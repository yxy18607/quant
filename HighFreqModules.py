# daily backtesting framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm
from datetime import datetime, timedelta
import os

MAX_INSTS = 10000

class BarDataLoader:
    def __init__(self, calendar: pd.Series, startdate: str, load_unit: str='60m', load_num: int=10, ):
        self.calendar = calendar
        self.startdate = startdate
        self.load_unit = load_unit
        self.load_num = load_num
        self.start_time, self.end_time = self.time_itvs()
        self.data_initialize()

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
        MAX_DATES = int(np.ceil(self.load_num/(240/int(self.load_unit[:-1]))))
        date = self.get_offset_trd(self.startdate, -MAX_DATES)
        self.dlf = {}
        while date < self.startdate:
            self.dlf[date] = pl.scan_parquet(f'./etf_mbar/etf_{date}.parquet')
            date = self.get_offset_trd(date, 1)
        self.rlf = []; self.rret = []
        for dt, lf in self.dlf.items():
            for start, end in zip(self.start_time, self.end_time):
                t1 = datetime.strptime(start, '%H:%M').time()
                t2 = datetime.strptime(end, '%H:%M').time()
                new_lf = lf.filter((pl.col('time_id').dt.time()>=t1)&(pl.col('time_id').dt.time()<=t2))
                self.rlf.append(new_lf)
                self.rret.append(new_lf
                                 .group_by('code').agg((pl.col('close').cast(pl.Float64).last()/pl.col('pre').cast(pl.Float64).first()-1).alias('ret'))
                                 .with_columns(pl.lit(datetime.strptime(f'{dt} {end}', "%Y%m%d %H:%M")).alias('datetime'))
                                 .collect()
                                 )
                if len(self.rlf)>self.load_num:
                    self.rlf.pop(0)
    
    def data_update(self, didx, start, end):
        if self.start_time.index(start) == 0:
            # 当日开始滚动时, 更新交易日
            self.dlf[didx] = pl.scan_parquet(f'./etf_mbar/etf_{didx}.parquet')
            self.dlf.pop(list(self.dlf.keys())[0])
        t1 = datetime.strptime(start, '%H:%M').time()
        t2 = datetime.strptime(end, '%H:%M').time()
        new_lf = self.dlf[didx].filter((pl.col('time_id').dt.time()>=t1)&(pl.col('time_id').dt.time()<=t2))
        self.rlf.append(new_lf)
        self.rlf.pop(0)
        self.rret.append(new_lf
                         .group_by('code').agg((pl.col('close').cast(pl.Float64).last()/pl.col('pre').cast(pl.Float64).first()-1).alias('ret'))
                         .with_columns(pl.lit(datetime.strptime(f'{didx} {end}', "%Y%m%d %H:%M")).alias('datetime'))
                         .collect()
                         )

class BackTest:
    """
    cfg参数配置说明
    ----------------------------------------------------
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

        self.fee = cfg.get('fee') if cfg.get('fee') is not None else 0
        print(f'------------------------backtesting {self.signal_id}-----------------------')

    def backward(self):
        dl = BarDataLoader(calendar=self.calendar,
                           startdate=self.startdate,
                           load_unit=self.load_unit,
                           load_num=self.load_num)
        start_time, end_time = dl.start_time, dl.end_time
        factors = []
        pbar = tqdm(self.dateindex)
        for didx in pbar:
            pbar.set_description(f'calculating {didx}: ')
            for start, end in zip(start_time, end_time):
                rlf = pl.concat(dl.rlf).filter(pl.col('time_id')<pl.col('time_id').max()) # 保证信号可交易性, 提前一分钟发信号
                factors.append(self.calculate_factor(rlf)
                               .with_columns(pl.lit(datetime.strptime(f'{didx} {end}', "%Y%m%d %H:%M")).alias('datetime'))
                               .select(['code', 'datetime', 'factor'])
                               .collect()
                               ) # 截止start前一时刻的信号
                dl.data_update(didx, start, end)
        self.factor = pl.concat(factors).with_columns(pl.col('factor').fill_nan(None).alias('factor'))
        self.ret = pl.concat(dl.rret)

    def calculate_factor(self, rlf: pl.LazyFrame):
        pass

    def factor_to_position(self):
        pass

    def get_pnl(self):
        self.pos_hf = self.factor_to_position().select(['code', 'datetime', 'factor', pl.col('pos').fill_nan(None)])
        self.pnl_hf = self.pos_hf.join(self.ret, on=['code', 'datetime'], how='inner').with_columns(
            pl.col('pos').fill_null(strategy='forward').alias('pos')
            ).with_columns([
            pl.col('datetime').dt.date().alias('date'),
            pl.col('pos').diff().abs().over('code').alias('dpos')]).with_columns(
            (pl.col('pos')*pl.col('ret')-pl.col('dpos')*self.fee).alias('pnl'))
        self.pnl_d = self.pnl_hf.group_by(['date', 'code']).agg(
            [pl.col('pnl').sum(),
             pl.col('dpos').sum()]).sort(['date'])
        self.pnl = self.pnl_d.group_by('date').agg([pl.col('pnl').mean(), pl.col('dpos').mean()]).with_columns(
            pl.col('pnl').cum_sum().alias('nav')
        )
        try:
            self.pos_hf.write_ipc(f'./dump/{self.signal_id}.feather')
        except OSError:
            print('写入feather失败, 请确认是否已经存在内存映射.')

    def stats(self):
        self.pnl_pd = self.pnl.with_columns(pl.col('date').dt.year().alias('year')).to_pandas().set_index('date')
        # total
        ret = self.pnl_pd['pnl'].mean() * 250
        sp = self.pnl_pd['pnl'].mean() / self.pnl_pd['pnl'].std() * np.sqrt(250)
        self.pnl_pd['dd_T'] = self.pnl_pd['nav'].cummax() - self.pnl_pd['nav']
        mdd = self.pnl_pd['dd_T'].max()
        dd_end = self.pnl_pd['dd_T'].idxmax()
        dd_start = self.pnl_pd.loc[:dd_end, 'nav'].idxmax()
        winr = (self.pnl_pd['pnl']>0).mean()
        odd = -self.pnl_pd.loc[self.pnl_pd['pnl']>0, 'pnl'].mean() / self.pnl_pd.loc[self.pnl_pd['pnl']<0, 'pnl'].mean() 
        calmar = ret / mdd
        tvr = self.pnl_pd['dpos'].mean()*250

        # yearly
        gpnl = self.pnl_pd.groupby('year')
        self.pnl_pd['dd_y'] = gpnl['nav'].cummax() - self.pnl_pd['nav']
        ret_y = gpnl['pnl'].mean() * 250
        sp_y = gpnl['pnl'].mean() / gpnl['pnl'].std() * np.sqrt(250)
        mdd_y = gpnl['dd_y'].max()
        dd_end_y = gpnl['dd_y'].idxmax()
        dd_start_y = gpnl.apply(lambda x: x.loc[:dd_end_y[x['year'].values[0]], 'nav'].idxmax())
        winr_y = gpnl['pnl'].apply(lambda x: (x>0).mean())
        odd_y = -gpnl['pnl'].apply(lambda x: x[x>0].mean() / x[x<0].mean())
        calmar_y = ret_y / mdd_y
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
        self.ret_d = self.pnl_hf.group_by(['date', 'code']).agg(pl.col('ret').sum().alias('daily_ret')).group_by('date').agg(pl.col('daily_ret').mean())
        benchmark = self.ret_d.select(pl.col('daily_ret').cum_sum()).to_pandas()['daily_ret']
        plot_df = pd.DataFrame({'strategy': self.pnl_pd['nav'], 'benchmark': benchmark.values})
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
        self.pnl_pd['pnl'].to_pickle(f"./pnl/{self.signal_id}.pnl.pkl")