# daily backtesting framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm
from datetime import datetime, timedelta

MAX_INSTS = 10000

class BarDataLoader:
    def __init__(self, parquet_path: str, calendar: pd.Series, start_date: str, load_unit: str='60m', load_num: int=10, ):
        self.parquet_path = parquet_path
        self.calendar = calendar
        self.start_date = start_date
        self.load_unit = load_unit
        self.load_num = load_num

    
        

        

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.signal_id = cfg.get('signal_id')
        self.mode = cfg.get('mode') # 1: long-only 2: short-only 0: long-short

        self.calendar = pd.read_pickle('./data/calendar.pkl')
        self.dateindex = self.calendar[(self.calendar>=self.startdate)&(self.calendar<=self.enddate)]
        self.load_unit = cfg.get('load_unit') # 'Xm'
        self.load_num = cfg.get('load_num')


        # self.signal_df = pd.read_pickle(f'./dump/{self.signal_id}.pkl').loc[self.dateindex]
        # self.listdays = pd.read_pickle('./data/listdays.pkl').loc[self.dateindex]
        # self.suspend = pd.read_pickle('./data/is_suspend.pkl').loc[self.dateindex]
        # self.st = pd.read_pickle('./data/is_st.pkl').loc[self.dateindex]
        # self.open = pd.read_pickle('./data/adjopen.pkl').loc[self.dateindex]
        # self.close = pd.read_pickle('./data/close.pkl').loc[self.dateindex]
        # self.limit = pd.read_pickle('./data/limit.pkl').loc[self.dateindex]
        # self.stopping = pd.read_pickle('./data/stopping.pkl').loc[self.dateindex]
        # self.limit_state = self.limit == self.open
        # self.stop_state = self.stopping == self.open

        self.get_valid()
        
        self.fee = cfg.get('fee') if cfg.get('fee') is not None else 0
        print(f'------------------------backtesting {self.signal_id}-----------------------')

    def time_itvs(self):
        start_time = []; end_time = []
        start = '09:30'
        period = self.load_unit
        ts1 = start
        while ts1<'15:00':
            start_time.append(ts1)
            ts2 = datetime.strptime(ts1, '%H:%M')+timedelta(minutes=int(period[:-1])-1)
            if ts1 == '09:30':
                ts2 += timedelta(minutes=1)
            end_time.append(ts2.strftime('%H:%M'))
            ts1 = (ts2+timedelta(minutes=1)).strftime('%H:%M')
            if ts1 == '11:31':
                ts1 = '13:01'
        return start_time, end_time

    def get_offset_trd(self, date, offset):
        return self.calendar[np.maximum(self.calendar.searchsorted(date)+offset, 0)]

    def data_initialize(self):
        MAX_DATES = int(np.ceil(self.load_num/(240/int(self.load_unit[:-1]))))+1
        date = self.get_offset_trd(self.startdate, -MAX_DATES)
        self.lf = {}
        while date != self.startdate:
            self.lf[date] = pl.scan_parquet(f'./etf_mbar/etf_{date}.parquet')
            date = self.get_offset_trd(date, 1)
        start_time, end_time = self.time_itvs()
        self.rolling_lf = []; self.itv_rets = []
        for dt, lf in self.lf.items():
            for start, end in zip(start_time, end_time):
                start = datetime.strptime(start, '%H:%M').time()
                end = datetime.strptime(end, '%H:%M').time()
                new_lf = lf.filter((pl.col('datetime').dt.time()>=start)&(pl.col('datetime').dt.time()<=end))
                self.rolling_lf.append(new_lf)
                self.itv_rets.append(new_lf.group_by('code').agg((pl.col('close').last()/pl.col('pre').first()-1).alias('ret')))
                if len(self.rolling_lf)>self.load_num:
                    self.rolling_lf.pop(0)
    
    def data_update(self, didx, start, end):
        new_lf = self.lf[didx].filter((pl.col('datetime').dt.time()>=start)&(pl.col('datetime').dt.time()<=end))
        self.rolling_lf.append(new_lf)
        self.rolling_lf.pop(0)
        self.itv_rets.append(new_lf.group_by('code').agg((pl.col('close').last()/pl.col('pre').first()-1).alias('ret')))
    
    def test(self):
        start_time, end_time = self.time_itvs()
        factors = []; rets = []
        for didx in self.dateindex:
            for start, end in zip(start_time, end_time):
                itv_lf = pl.concat(self.rolling_lf)
                factors.append(self.calculate_factor(itv_lf).collect()) # 截止start前一时刻的信号
                if start_time.index(start) == 0:
                    # 当日开始滚动, 需读取当日数据, 并剔除最旧日数据
                    self.lf[didx] = pl.scan_parquet(f'./etf_mbar/etf_{didx}.parquet')
                    self.lf.pop(list(self.lf.keys())[0])
                self.data_update(didx, start, end)

    def calculate_factor(self, itv_lf):
        pass

    def get_position(self):
        self.pos_df = self.signal_df.shift(1) # 收盘发信号，次日开盘生成仓位
        # 涨停无法买入、跌停无法卖出、停牌无法交易
        limit_pos = self.limit_state & (self.pos_df.diff()>0)
        stop_pos = self.stop_state & (self.pos_df.diff()<0)
        self.pos_df[limit_pos|stop_pos|self.suspend] = np.nan
        self.pos_df.where(~(limit_pos|stop_pos|self.suspend), self.pos_df.ffill(), inplace=True)
        self.pos_df = self.pos_df.shift(1) # T+2获得持仓收益

        if self.mode == 1:
            self.pos_df = np.maximum(0, self.pos_df)
        elif self.mode == 2:
            self.pos_df = np.minimum(0, self.pos_df)

        self.dpos_df = self.pos_df.fillna(0).diff() # 计算调仓幅度需剔除nan的影响
        self.dpos_df[self.pos_df.isna()] = np.nan
        self.returns = self.open.pct_change(fill_method=None).fillna(0)

    def generate_pnl(self):
        pnl = self.pos_df.mul(self.returns)
        pnl = pnl - self.dpos_df.abs() * self.fee
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

    def save_pnl(self):
        self.pnl['pnl'].to_pickle(f"./pnl/{self.signal_id}.pnl.pkl")

class Beta:
    """
    cfg参数配置说明
    --------------------------------------------------
    key             value           comment

    startdate       str             信号计算起始日
    enddate         str             信号计算终止日
    信号计算默认多回溯120天数据, 防止出现前几个交易日无信号的情况
    output: 
    self.beta - betaname_r.pkl 因子原始值
    self.df_signal - betaname.pkl 因子仓位值
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
        
    def hf_to_daily(self):
        daily_df = []
        start = datetime.now()
        pbar = tqdm(self.dateindex, unit='day')
        for didx in pbar:
            df_lazy = pl.scan_parquet(f'./stock_mbar/ashare_{didx}.parquet')
            daily_lazy = self.generate_daily(df_lazy) # output: code, factor
            daily_lazy = daily_lazy.with_columns(pl.lit(didx).alias('trade_date'))
            daily_df.append(daily_lazy.collect())
            pbar.set_description(f'executing on {didx[:-2]} | cost {str(datetime.now()-start).split('.')[0]}')
        daily_df = pl.concat(daily_df).pivot(on='code', index='trade_date', values='factor').to_pandas().set_index('trade_date')
        columns = pd.read_pickle('./data/close.pkl').columns
        self.daily_df = daily_df.reindex(columns=columns)

    def generate_daily(self, df_lazy: pl.LazyFrame):
        "UNIMPLEMENTED"
        pass

    def generate_signal(self):
        pass

    def dump(self):
        self.beta.to_pickle(f'./dump/{self.__class__.__name__}_r.pkl')
        self.signal_df.to_pickle(f'./dump/{self.__class__.__name__}.pkl')

    