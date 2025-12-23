import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
from dataclasses import dataclass
from typing import Union

@dataclass
class DailyCTA:
    """
    cfg参数配置说明
    -----------------------------------------------------
    key             value           comment

    startdate       str             回测起始日
    enddate         str             回测终止日
    zone            str             交易市场标注
    slippage        float64         回测绝对滑点值
    fee             float64         回测百分比费率
    trade_price     str             open-按日次开盘价撮合; close-按当日收盘价撮合; vwap-按次日均价撮合
    signal_id       str             读取的信号名称(包含多品种的信号)
    instruments     list, dict      回测的交易品种列表, list-发信号和交易的品种一一对应, dict-发信号和交易的是不同的品种, 由key和value确定对应关系
    mode            int             0-多空信号; 1-纯多头信号; 2-纯空头信号
    period          str             绩效统计周期, Y-yearly; M-monthly, 默认Y
    dump_pnl        bool            True-缓存pnl; False-不缓存
    """
    startdate: str
    enddate: str
    signal_id: str
    instruments: Union[list, dict]
    trade_price: str = 'open'
    zone: str = ''
    mode: int = 0
    slippage: float = 0.
    fee: float = 0.
    pnl_period: str = 'Y'
    dump_pnl: bool = True
    update: bool = False

    def __post_init__(self):
        if len(self.zone)>0:
            self.zone = f'_{self.zone}'
        calendar = pd.read_pickle(f'./data/calendar{self.zone}.pkl')
        self.signal_df = pd.read_pickle(f'./dump/{self.signal_id}.pkl')
        self.check_calendar(calendar)
        self.trading_insts()
        print(f"--------------backtesting {self.signal_id}--------------")

    def __call__(self):
        self.get_pos()
        self.get_pnl()
        self.stat()
    
    def check_calendar(self, calendar):
        if self.trade_price in ['open', 'vwap']:
            offset = 3
        elif self.trade_price == 'close':
            offset = 2
        startdate = calendar.iloc[calendar.searchsorted(self.startdate, 'left')-offset]
        dateindex = self.signal_df.index.intersection(calendar)
        self.dateindex = pd.Series(dateindex[(dateindex>=startdate)&(dateindex<=self.enddate)])
    
    def trading_insts(self):
        if isinstance(self.instruments, dict):
            signal_col = list(self.instruments.keys())
            trade_col = list(self.instruments.values())
            self.instruments = trade_col
            self.signal_df = self.signal_df.loc[self.dateindex, signal_col]
            self.signal_df.columns = trade_col
        else:
            self.signal_df = self.signal_df.loc[self.dateindex, self.instruments]

    def profit_ana(self, retdays=10):
        pred_corr = pd.Series()
        for inst in self.data_dict.keys():
            ret = self.data_dict[inst]['returns'].rolling(retdays).sum().shift(retdays-1)
            pos = self.pos[inst]
            pred_corr[inst] = pos.corr(ret)
        print(pred_corr)

    def get_pos(self):
        self.data_dict = {}
        for inst in self.instruments:
            df = pd.read_pickle(f'./data/{inst}.pkl').loc[self.dateindex]
            if self.trade_price == 'open':
                df['returns'] = df['open'].pct_change().fillna(0)
            elif self.trade_price == 'close':
                df['returns'] = df['close'].pct_change().fillna(0)
            elif self.trade_price == 'vwap':
                df['vwap'] = df['amount'] / df['volume']
                df['returns'] = df['vwap'].pct_change().fillna(0)
            self.data_dict[inst] = df

        if self.trade_price in ['open', 'vwap']:
            self.pos = self.signal_df.shift(2).fillna(0)
        elif self.trade_price == 'close':
            self.pos = self.signal_df.shift(1).fillna(0)
        if self.mode == 1:
            self.pos = np.maximum(0, self.pos)
        elif self.mode == 2:
            self.pos = np.minimum(0, self.pos)

    def get_pnl(self):
        pnl = pd.DataFrame()
        for inst in self.data_dict.keys():
            df = self.data_dict[inst]
            pnl[inst] = df['returns'] * self.pos[inst]
            pnl[inst] = pnl[inst] - self.pos[inst].diff(1).abs() * (self.fee + self.slippage / df['close'])
        pnl = pnl.mean(1)
        self.pnl = pd.DataFrame({'pnl': pnl,
                                 'dpos': self.pos.diff().abs().mean(1)})
        self.pnl = self.pnl[self.pnl.index>=self.startdate]
        if self.dump_pnl:
            if self.update:
                old_pnl = pd.read_pickle(f"./pnl/{self.signal_id}.pnl.pkl")
                new_pnl = self.pnl.loc[self.pnl.index>old_pnl.index[-1]]
                pd.concat([old_pnl, new_pnl], axis=0).to_pickle(f"./pnl/{self.signal_id}.pnl.pkl")
            else:
                self.pnl.to_pickle(f"./pnl/{self.signal_id}.pnl.pkl")

    def stat(self):
        self.pnl['nav'] = self.pnl['pnl'].cumsum()
        pnl = self.pnl['pnl']; nav = self.pnl['nav']
        # total
        ret = pnl.mean()*250
        sp = pnl.mean() / pnl.std() * np.sqrt(250)
        dd_T = nav.cummax() - nav
        mdd = dd_T.max()
        dd_end = dd_T.idxmax()
        dd_start = nav[:dd_end].idxmax()
        dd_duration = np.bincount(nav.cummax().diff().astype(bool).cumsum()).max()
        profit = pnl[pnl>0]; loss = pnl[pnl<0]
        winr = len(profit)/(len(profit)+len(loss))
        odd = -profit.mean()/loss.mean()
        calmar = ret / mdd
        tvr = self.pnl['dpos'].mean(0)*250

        # by period
        if self.pnl_period == 'Y':
            group_col = 'year'
            self.pnl[group_col] = self.pnl.index.str[:4]
        elif self.pnl_period == 'M':
            group_col = 'month'
            self.pnl[group_col] = self.pnl.index.str[:6]
        elif self.pnl_period == 'H':
            group_col = 'half'
            self.pnl[group_col] = self.pnl.index.str[:4]+(self.pnl.index.str[4:6].astype(int)//7).astype(str)
        elif self.pnl_period == 'Q':
            group_col = 'quarter'
            self.pnl[group_col] = self.pnl.index.str[:4]+((self.pnl.index.str[4:6].astype(int)-1)//3).astype(str)
        gpnl = self.pnl.groupby(group_col)
        pnl_y = gpnl['pnl']; nav_y = gpnl['nav']
        ret_y = pnl_y.mean() * 250
        sp_y = pnl_y.mean() / pnl_y.std() * np.sqrt(250)
        mdd_y = nav_y.apply(lambda x: (x.cummax()-x).max())
        dd_end_y = nav_y.apply(lambda x: (x.cummax()-x).idxmax())
        dd_start_y = nav_y.apply(lambda x: x[:dd_end_y[x.name]].idxmax())
        dd_duration_y = nav_y.apply(lambda g: np.bincount(g.cummax().diff().astype(bool).cumsum()).max())
        winr_y = pnl_y.apply(lambda x: (x>0).sum()/((x>0).sum()+(x<0).sum()))
        odd_y = -pnl_y.apply(lambda x: x[x>0].mean() / x[x<0].mean())
        calmar_y = ret_y / mdd_y
        tvr_y = gpnl['dpos'].mean() * 250


        index1 = gpnl.head(1).index
        index2 = gpnl.tail(1).index
        out = pd.DataFrame({'ret': ret_y.values*100,
                            'sp': sp_y.values,
                            'dd': mdd_y.values*100,
                            'dd_start': dd_start_y.values,
                            'dd_end': dd_end_y.values,
                            'dd_duration': dd_duration_y.values,
                            'anntvr': tvr_y.values,
                            'winr': winr_y.values*100,
                            'odd': odd_y.values,
                            'calmar': calmar_y.values}, index=pd.MultiIndex.from_arrays([index1, index2], names=['from', 'to']))
        out.loc[(self.pnl.index[0], self.pnl.index[-1]), :] = [ret*100, sp, mdd*100, dd_start, dd_end, dd_duration, tvr, winr*100, odd, calmar]
        print(out)


    def plot_curve(self, os_startdate:str=None):
        benchmark = pd.DataFrame()
        for inst in self.data_dict.keys():
            benchmark[inst] = self.data_dict[inst]['returns']
        benchmark = benchmark[benchmark.index>=self.startdate].mean(1).cumsum()
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
    zone            str             交易市场, 不同市场有不同calendar, 默认为大陆市场, hk为港股, us为美股
    instruments     list            计算信号的品种列表
    """
    def __init__(self, cfg):
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.zone = '' if not cfg.get('zone') else f"_{cfg.get('zone')}" # hk或us市场的交易日历不同，要分开处理
        self.calendar = pd.read_pickle(f'./data/calendar{self.zone}.pkl')
        self.instruments = cfg.get('instruments')
        _startdate = self.calendar.searchsorted(self.startdate, 'left')-60
        _enddate = self.calendar.searchsorted(self.enddate, side='right')
        _mindate = self.calendar.searchsorted('20140101', 'left')

        self.dateindex = self.calendar.iloc[np.maximum(_startdate, _mindate):_enddate]
        self.signal_df = pd.DataFrame()

    def process(self):
        signal_df = pd.DataFrame()
        for inst in self.instruments:
            df = pd.read_pickle(f'./data/{inst}.pkl').loc[self.dateindex]
            if not self.intra_to_daily.__doc__ == 'UNIMPLEMENTED':
                pf = (pl.scan_parquet(f'./idx_mbar/{inst}.parquet')
                      .filter(pl.col('date_id').is_in(pl.Series(pd.to_datetime(self.dateindex).dt.date)))
                    .with_columns([pl.col('close').cast(pl.Float64),
                                    pl.col('open').cast(pl.Float64),
                                    pl.col('high').cast(pl.Float64),
                                    pl.col('low').cast(pl.Float64),
                                    pl.col('pre').cast(pl.Float64),
                                    pl.col('volume').cast(pl.Int64),
                                    pl.col('amount').cast(pl.Int64)]))
                pf = self.intra_to_daily(pf).sort('date_id').collect().to_pandas().set_index('date_id')
                pf.index = pf.index.strftime('%Y%m%d')
                df = pd.concat([df, pf], axis=1, join='inner')
            signal = self.generate_signal(df)
            signal_df[inst] = signal
        signal_df.index = self.dateindex
        self.signal_df = signal_df

    def generate_signal(self, df: pd.DataFrame):
        pass

    def intra_to_daily(self, pf: pl.LazyFrame):
        "UNIMPLEMENTED"

    def dump(self):
        self.signal_df.to_pickle(f'./dump/{self.__class__.__name__}.pkl')

