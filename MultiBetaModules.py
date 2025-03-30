# daily backtesting framework

import pandas as pd
import numpy as np

MAX_INSTS = 10000

class BackTest:
    """
    for in-sample and out-sample cross-sectional alpha backtest
    # inputs: daily signal, stock_info, prices, cfg
    # outputs: backtest stats
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.signal_id = cfg.get('signal_id')
        self.mode = cfg.get('mode') # 1: long-only 2: short-only 0: long-short

        self.calendar = pd.read_pickle('./data/calendar.pkl')
        self.dateindex = self.calendar[(self.calendar>=self.startdate)&(self.calendar<=self.enddate)]

        self.signal_df = pd.read_pickle(f'./dump/{self.signal_id}.pkl') # 这里的信号是因子转换成仓位后的结果
        self.listdays = pd.read_pickle('./data/listdays.pkl').loc[self.dateindex]
        self.suspend = pd.read_pickle('./data/is_suspend.pkl').loc[self.dateindex]
        self.st = pd.read_pickle('./data/is_st.pkl').loc[self.dateindex]
        self.open = pd.read_pickle('./data/open.pkl').loc[self.dateindex]
        self.close = pd.read_pickle('./data/close.pkl').loc[self.dateindex]
        self.limit = pd.read_pickle('./data/limit.pkl').loc[self.dateindex]
        self.stopping = pd.read_pickle('./data/stopping.pkl').loc[self.dateindex]
        self.limit_state = self.limit == self.open
        self.stop_state = self.stopping == self.open

        self.get_valid()
        
        self.fee = cfg.get('fee')
        print(f'------------------------backtesting {self.signal_id}-----------------------')

    def __call__(self):
        self.get_position()
        self.beta_ana()
        self.generate_pnl()
        self.stats()
    
    def get_valid(self):
        lp = self.close > 1 # exclude stocks whose price <= 1
        st = ~self.st # exclude st stocks
        rl = self.listdays >= 300 # exclude recently listed stocks and delisted stocks
        suspend = ~self.suspend # exclude suspended stocks
        bj = ~self.close.columns.str.contains('BJ')
        bj = pd.DataFrame(np.tile(bj, (len(self.close.index), 1)), index=self.close.index, columns=self.price.columns)
        valid = lp & st & rl & suspend & bj
        self.valid = valid
        self.signal_df.where(valid, np.nan, inplace=True)

    def beta_ana(self, period=5):
        fret = self.open.pct_change(period)
        signal = self.signal_df.shift(period+1)
        ic = fret.corrwith(signal, axis=1)
        rollingic = fret.rolling(120).corr(signal).mean(1).dropna()
        print(f"history ic: {ic.mean()}")
        print(f"rolling ic: {rollingic}")
        rollingic.index = pd.to_datetime(rollingic.index)
        rollingic.plot()

    def get_position(self):
        self.pos_df = self.signal_df.shift(1) # 收盘发信号，次日开盘生成仓位
        # 涨停无法买入、跌停无法卖出、停牌无法交易
        limit_pos = self.limit_state & self.pos_df.diff()>0
        stop_pos = self.stop_state & self.pos_df.diff()<0
        self.pos_df[limit_pos|stop_pos|self.suspend] = np.nan
        self.pos_df.where(~(limit_pos|stop_pos|self.suspend), self.pos_df.ffill(), inplace=True)
        self.pos_df = self.pos_df.shift(1) # T+2获得持仓收益

        if self.mode == 1:
            self.pos_df = np.maximum(0, self.pos_df)
        elif self.mode == 2:
            self.pos_df = np.minimum(0, self.pos_df)

        self.dpos_df = self.pos_df.fillna(0).diff() # 计算调仓幅度需剔除nan的影响
        self.dpos_df[self.pos_df.isna()] = np.nan
        self.returns = self.open.pct_change().fillna(0)

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

class Beta:
    def __init__(self, cfg):
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        self.combo = cfg.get('combo') # bool
        self.calendar = pd.read_pickle('./data/calendar.pkl')
        if not self.combo:
            _startdate = self.calendar.searchsorted(self.startdate, 'left')-250+1
        else:
            _startdate = self.calendar.searchsorted(self.startdate, 'left')
        _enddate = self.calendar.searchsorted(self.enddate, side='right')
        _mindate = self.calendar.searchsorted('20140101', 'left')

        self.dateindex = self.calendar.iloc[np.maximum(_startdate, _mindate):_enddate]
        self.signal_df = pd.DataFrame()
        
    def hf_to_daily(self):
        for didx in self.dateindex:
            daily = self.generate_daily(didx)
            self.signal_df.loc[didx] = daily

    def generate_daily(self, didx):
        pass

    def generate_beta(self):
        pass

    def dump(self):
        # self.alpha_df = self.alpha_df.loc[self.alpha_df.index>=self.startdate]
        self.signal_df.to_pickle(f'./dump/{self.__class__.__name__}.pkl')

    