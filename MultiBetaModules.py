# daily backtesting framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from typing import Union

MAX_INSTS = 10000

@dataclass
class BackTest:
    """
    cfg参数配置说明
    -----------------------------------------------------
    key             value           comment

    startdate       str             回测起始日
    enddate         str             回测终止日
    slippage        float64         回测绝对滑点值
    fee             float64         回测百分比费率
    signal_id       str             读取的信号名称(包含多品种的信号)
    mode            int             0-多空信号; 1-纯多头信号; 2-纯空头信号
    """
    startdate: str
    enddate: str
    signal_id: str
    instruments: Union[list, pd.DataFrame] = None # 品种池名称/列表
    mode: int = 0 # 1: long-only 2: short-only 0: long-short
    category: str = 'stock'
    fee: np.float64 = 0.
    dump_pnl: bool = True
    
    def __post_init__(self):
        self.calendar = pd.read_pickle('./data/calendar.pkl')
        self.dateindex = self.calendar[(self.calendar>=self.startdate)&(self.calendar<=self.enddate)]
        self.signal_df = pd.read_pickle(f'./dump/{self.signal_id}.pkl').loc[self.dateindex]

        if self.category == 'stock':
            self.listdays = pd.read_pickle('./data/listdays.pkl').loc[self.dateindex]
            self.suspend = pd.read_pickle('./data/is_suspend.pkl').loc[self.dateindex]
            self.st = pd.read_pickle('./data/is_st.pkl').loc[self.dateindex]
            self.open = pd.read_pickle('./data/adjopen.pkl').loc[self.dateindex]
            self.close = pd.read_pickle('./data/close.pkl').loc[self.dateindex]
            self.limit = pd.read_pickle('./data/limit.pkl').loc[self.dateindex]
            self.stopping = pd.read_pickle('./data/stopping.pkl').loc[self.dateindex]
            self.limit_state = self.limit == self.open
            self.stop_state = self.stopping == self.open
        elif self.category == 'etf':
            self.open = pd.read_pickle('./data/etf_open.pkl').loc[self.dateindex]
            self.close = pd.read_pickle('./data/etf_close.pkl').loc[self.dateindex]

        self.get_valid()
        print(f'------------------------backtesting {self.signal_id}-----------------------')

    def __call__(self):
        self.get_position()
        self.generate_pnl()
        self.stats()
    
    def get_valid(self):
        if self.category == 'stock':
            lp = self.close > 1 # exclude stocks whose price <= 1
            st = ~self.st # exclude st stocks
            rl = self.listdays >= 300 # exclude recently listed stocks and delisted stocks
            bj = ~self.close.columns.str.contains('BJ') # 剔除北交所
            bj = pd.DataFrame(np.tile(bj, (len(self.close.index), 1)), index=self.close.index, columns=self.close.columns)
            valid = lp & st & rl & bj
        elif self.category == 'etf':
            valid = pd.DataFrame(True, index=self.close.index, columns=self.close.columns)

        if self.instruments is not None:
            if isinstance(self.instruments, pd.DataFrame):
                valid = valid & self.instruments
            elif isinstance(self.instruments, list):
                valid[~valid.columns.isin(self.instruments)] = False
        self.signal_df.where(valid, np.nan, inplace=True)
        self.valid = valid

    def beta_ana(self, period=1):
        fret = self.open.pct_change(period, fill_method=None)
        signal = self.signal_df.shift(period+1)
        rollingic = fret.rolling(120).corr(signal).mean(1).dropna()
        cumic = rollingic.cumsum()
        print(f"avg ic: {rollingic.median()}")
        cumic.index = pd.to_datetime(cumic.index)
        cumic.plot()
        plt.show()

    def get_position(self):
        self.pos_df = self.signal_df.shift(1) # 收盘发信号，次日开盘生成仓位
        if self.mode == 1:
            self.pos_df = np.maximum(0, self.pos_df)
        elif self.mode == 2:
            self.pos_df = np.minimum(0, self.pos_df)
        if self.category == 'stock':
            # 先处理因子输出nan(包括未上市/已退市/被st/因子值nan/停牌)导致dpos无法计算的问题
            dpos_before = self.pos_df.diff()
            self.pos_df.fillna(0, inplace=True) # 防止后续fillna时覆盖真实的缺失样本
            # 停牌时延续的仓位可能影响复牌时调仓方向, 因此优先处理停牌仓位
            suspend = pd.read_pickle('./data/is_suspend.pkl').loc[self.dateindex]
            self.pos_df[suspend] = np.nan
            self.pos_df.where(~self.valid.shift(1, fill_value=False), self.pos_df.ffill(), inplace=True)
            # 涨停无法买入, 跌停无法卖出
            limit_pos = self.limit_state & (self.pos_df.diff()>0)
            stop_pos = self.stop_state & (self.pos_df.diff()<0)
            self.pos_df[limit_pos|stop_pos] = np.nan
            self.pos_df.where(~(limit_pos|stop_pos), self.pos_df.ffill(), inplace=True)
            dpos_after = self.pos_df.diff()
            # 无效仓位条件: 非停牌/涨跌停/因子值本身导致的不变仓位
            condition_nan = (dpos_after==0)&(dpos_before.isna())&(~suspend)&(~limit_pos)&(~stop_pos)
            self.pos_df[condition_nan] = np.nan
        self.pos_df = self.pos_df.shift(1) # T+2获取收益
        self.dpos_df = self.pos_df.diff()
        self.returns = self.open.pct_change(fill_method=None).fillna(0)

    def generate_pnl(self):
        pnl = self.pos_df.mul(self.returns)
        pnl = pnl - self.dpos_df.abs() * self.fee
        self.pnl_df = pnl.copy()
        pnl = pnl.mean(1)
        self.pnl = pd.DataFrame({'pnl': pnl, 'nav': pnl.cumsum()})

    def stats(self):
        self.pnl['year'] = self.pnl.index.str[:4]
        # total
        ret = self.pnl['pnl'].mean() * 250
        sp = self.pnl['pnl'].mean() / self.pnl['pnl'].std() * np.sqrt(250)
        self.pnl['dd_T'] = self.pnl['nav'].cummax() - self.pnl['nav']
        mdd = self.pnl['dd_T'].max()
        dd_end = self.pnl['dd_T'].idxmax()
        dd_start = self.pnl.loc[:dd_end, 'nav'].idxmax()
        winr = (self.pnl['pnl']>0).mean()
        odd = -self.pnl.loc[self.pnl['pnl']>0, 'pnl'].mean() / self.pnl.loc[self.pnl['pnl']<0, 'pnl'].mean() 
        calmar = ret / mdd
        tvr = self.dpos_df.abs().mean(1).mean(0)*250

        # yearly
        gpnl = self.pnl.groupby('year')
        self.pnl['dd_y'] = gpnl['nav'].cummax() - self.pnl['nav']
        ret_y = gpnl['pnl'].mean() * 250
        sp_y = gpnl['pnl'].mean() / gpnl['pnl'].std() * np.sqrt(250)
        mdd_y = gpnl['dd_y'].max()
        dd_end_y = gpnl['dd_y'].idxmax()
        dd_start_y = gpnl.apply(lambda x: x.loc[:dd_end_y[x['year'].values[0]], 'nav'].idxmax())
        winr_y = gpnl['pnl'].apply(lambda x: (x>0).mean())
        odd_y = -gpnl['pnl'].apply(lambda x: x[x>0].mean() / x[x<0].mean())
        calmar_y = ret_y / mdd_y
        self.pnl['dpos'] = self.dpos_df.abs().mean(1)
        tvr_y = gpnl['dpos'].mean() * 250

        index1 = gpnl.head(1).index
        index2 = gpnl.tail(1).index
        out = pd.DataFrame({'ret': ret_y.values*100,
                            'sp': sp_y.values,
                            'dd': mdd_y.values*100,
                            'dd_start': dd_start_y.values,
                            'dd_end': dd_end_y.values,
                            'anntvr': tvr_y.values,
                            'winr': winr_y.values*100,
                            'odd': odd_y.values,
                            'calmar': calmar_y.values}, index=pd.MultiIndex.from_arrays([index1, index2], names=['from', 'to']))
        out.loc[(self.dateindex.iloc[0], self.dateindex.iloc[-1]), :] = [ret*100, sp, mdd*100, dd_start, dd_end, tvr, winr*100, odd, calmar]
        print(out)
        if self.dump_pnl:
            self.pnl[['pnl', 'dpos']].to_pickle(f"./pnl/{self.signal_id}.pnl.pkl")
        
    def stats_cross(self):
        # 统计每个品种表现
        annret = self.pnl_df.mean(0)*250
        annsp = self.pnl_df.mean(0)/self.pnl_df.std(0) * np.sqrt(250)
        nav = self.pnl_df.cumsum(0)
        mdd = (nav.cummax()-nav).max()
        tvr = self.dpos_df.abs().mean(0)*250
        winr = (self.pnl_df>0).sum()/self.pnl_df.notna().sum()
        pnl_win = self.pnl_df[self.pnl_df>0].mean()
        pnl_loss = self.pnl_df[self.pnl_df<0].mean()
        odd = -pnl_win/pnl_loss
        calmar = annret/mdd
        sample = self.pnl_df.notna().mean()
        dateitv = self.pnl_df.apply(lambda x: f'{x.first_valid_index()}-{x.last_valid_index()}')
        out = pd.concat([annret, annsp, mdd, tvr, winr, odd, calmar, sample, dateitv], axis=1)
        out.columns = ['annret', 'annsp', 'mdd', 'tvr', 'winr', 'odd', 'calmar', 'sample', 'dateitv']
        out.to_pickle(f"./pnl/{self.signal_id}.stats_cross.pkl")


    def plot_curve(self, os_startdate:str=None):
        benchmark = self.returns.mean(1).cumsum()
        plot_df = pd.DataFrame({'strategy': self.pnl['nav'], 'benchmark': benchmark})
        plot_df.index = pd.to_datetime(plot_df.index)
        if not os_startdate:
            plot_df.plot()
        else:
            plot_df['is'] = plot_df.index<pd.to_datetime(os_startdate)
            ax = plot_df.loc[plot_df['is'], 'strategy'].plot(color='#1f77b4', label='strategy(IS)', figsize=(12, 6))
            plot_df.loc[plot_df['is'], 'benchmark'].plot(ax=ax, color='#ffbb78', label='benchmark(IS)')
            plot_df.loc[~plot_df['is'], 'strategy'].plot(ax=ax, color='#d62728', label='strategy(OS)')
            plot_df.loc[~plot_df['is'], 'benchmark'].plot(ax=ax, color='#ff7f0e', label='benchmark(OS)')
            ax.axvline(pd.to_datetime(os_startdate), color="black", linestyle="--")  # 添加分割线
        plt.legend()
        plt.show()

class Beta:
    """
    cfg参数配置说明
    --------------------------------------------------
    key             value           comment

    startdate       str             信号计算起始日
    enddate         str             信号计算终止日
    信号计算默认多回溯120天数据, 防止出现前几个交易日无信号的情况
    output: 
    self.factor - factorname_r.pkl 因子原始值
    self.df_signal - factorname.pkl 因子仓位值
    """
    def __init__(self, cfg):
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.calendar = pd.read_pickle('./data/calendar.pkl')
        _startdate = self.calendar.searchsorted(self.startdate, 'left')-120
        _enddate = self.calendar.searchsorted(self.enddate, side='right')
        _mindate = self.calendar.searchsorted('20140101', 'left')

        self.dateindex = self.calendar.iloc[np.maximum(_startdate, _mindate):_enddate]
        self.signal_df = pd.DataFrame()
        
    def process(self):
        daily_df = []
        start = datetime.now()
        pbar = tqdm(self.dateindex, unit='day')
        for didx in pbar:
            df_lazy = pl.scan_parquet(f'./stock_mbar/ashare_{didx}.parquet')
            df_lazy = df_lazy.with_columns([pl.col('close').cast(pl.Float64).alias('close'),
                                            pl.col('open').cast(pl.Float64).alias('open'),
                                            pl.col('high').cast(pl.Float64).alias('high'),
                                            pl.col('low').cast(pl.Float64).alias('low'),
                                            pl.col('pre').cast(pl.Float64).alias('pre'),
                                            pl.col('volume').cast(pl.Int64).alias('volume'),
                                            pl.col('amount').cast(pl.Int64).alias('amount')])
            daily_lazy = self.intra_to_daily(df_lazy) # output: code, factor
            daily_lazy = daily_lazy.with_columns(pl.lit(didx).alias('trade_date'))
            daily_df.append(daily_lazy.collect())
            pbar.set_description(f'executing on {didx[:-2]} | cost {str(datetime.now()-start).split('.')[0]}')
        daily_df = pl.concat(daily_df).pivot(on='code', index='trade_date', values='factor').to_pandas().set_index('trade_date')
        columns = pd.read_pickle('./data/close.pkl').columns
        self.daily_df = daily_df.reindex(columns=columns)

    def intra_to_daily(self, df_lazy: pl.LazyFrame):
        "UNIMPLEMENTED"
        pass

    def generate_signal(self):
        pass

    def dump(self):
        self.factor.to_pickle(f'./dump/{self.__class__.__name__}_r.pkl')
        self.signal_df.to_pickle(f'./dump/{self.__class__.__name__}.pkl')

    