import numpy as np
import pandas as pd

class DailyCTA:
    def __init__(self, cfg):
        self.startdate = cfg.get('startdate')
        self.enddate = cfg.get('enddate')
        calendar = pd.read_pickle('./data/calendar.pkl')
        self.dateindex = calendar[(calendar>=self.startdate)&(calendar<=self.enddate)]
        self.slippage = cfg.get('slippage')
        self.fee = cfg.get('fee')
        trade_price = cfg.get('trade_price') # ['close', 'open', 'vwap']
        signal_id = cfg.get('signal_id')
        instruments = cfg.get('instruments') # list

        self.df_signal = pd.read_pickle(f'./dump/{signal_id}.pkl').loc[self.dateindex] # index = trade_date, columns=instruments, values=signals
        self.data_dict = {}
        for inst in instruments:
            df = pd.read_pickle(f'./data/{inst}.pkl').loc[self.dateindex]
            if trade_price == 'open':
                df['returns'] = df['open'].pct_change()
                self.pos = self.df_signal.shift(2).fillna(0)
            elif trade_price == 'close':
                df['returns'] = df['pct_change']/100
                self.pos = self.df_signal.shift(1).fillna(0)
            elif trade_price == 'vwap':
                df['vwap'] = df['amount'] / df['volume']
                df['returns'] = df['vwap'].pct_change()
                self.pos = self.df_signal.shift(2).fillna(0)
            self.data_dict[inst] = df

    def __call__(self):
        self.get_pnl()
        self.stat()

    def profit_ana(self, retdays=10):
        pred_corr = pd.Series()
        for inst in self.data_dict.keys():
            ret = self.data_dict[inst]['returns'].rolling(retdays).sum().shift(retdays-1)
            pos = self.pos[inst]
            pred_corr[inst] = pos.corr(ret)
        print(pred_corr)

    def get_pnl(self):
        pnl = pd.DataFrame()
        for inst in self.data_dict.keys():
            df = self.data_dict[inst]
            pnl[inst] = df['returns'] * self.pos[inst]
            pnl[inst] = pnl[inst] - self.pos[inst].diff(1).abs() * (self.fee + self.slippage / df['close'])
        pnl = pnl.mean(1)
        self.pnl = pd.DataFrame({'pnl': pnl})

    def stat(self):
        self.pnl['year'] = self.pnl.index.str[:4]
        self.pnl['nav'] = self.pnl['pnl'].cumsum()
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
        tvr = self.pos.diff(1).abs().mean(1).mean(0)*250

        # yearly
        gpnl = self.pnl.groupby('year')
        self.pnl['dd_y'] = gpnl['nav'].cummax() - self.pnl['nav']
        ret_y = gpnl['pnl'].mean() * 250
        sp_y = gpnl['pnl'].mean() / gpnl['pnl'].std() * np.sqrt(250)
        mdd_y = gpnl['dd_y'].max()
        dd_end_y = gpnl['dd_y'].idxmax()
        dd_start_y = gpnl.apply(lambda x: x.loc[:dd_end_y[x['year'].values[0]], 'nav'].idxmax())
        winr_y = gpnl['pnl'].apply(lambda x: (x>0).mean())
        odd_y = gpnl['pnl'].apply(lambda x: -x[x>0].mean() / x[x<0].mean())
        calmar_y = ret_y / mdd_y
        self.pnl['dpos'] = self.pos.diff(1).abs().mean(1)
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
        self.calendar = pd.read_pickle('./data/calendar.pkl')
        self.instruments = cfg.get('instruments')
        _startdate = self.calendar.searchsorted(self.startdate, 'left')
        _enddate = self.calendar.searchsorted(self.enddate, side='right')
        _mindate = self.calendar.searchsorted('20140101', 'left')

        self.dateindex = self.calendar.iloc[np.maximum(_startdate, _mindate):_enddate]
        self.signal_df = pd.DataFrame()

    def __call__(self):
        self.generate_signal()
        self.dump()

    def generate_signal(self):
        signal_df = pd.DataFrame()
        for inst in self.instruments:
            df = pd.read_pickle(f'./data/{inst}.pkl').loc[self.dateindex]
            signal = self.generate_beta(df)
            signal_df[inst] = signal
        self.signal_df = signal_df

    def generate_beta(self, df):
        pass

    def dump(self):
        self.signal_df.to_pickle(f'./dump/{self.__class__.__name__}.pkl')

