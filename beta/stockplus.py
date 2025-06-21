import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from MultiBetaModules import Beta

class stockplus(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.var1 = pd.read_pickle('./dump/stocktiming1.pkl').loc[self.dateindex]
        self.var2 = pd.read_pickle('./dump/stocktiming2.pkl').loc[self.dateindex]
        self.var3 = pd.read_pickle('./dump/stocktiming3.pkl').loc[self.dateindex]
        self.var4 = pd.read_pickle('./dump/stocktiming4.pkl').loc[self.dateindex]
        self.var5 = pd.read_pickle('./dump/stocktiming5.pkl').loc[self.dateindex]
        self.var6 = pd.read_pickle('./dump/stocktiming6.pkl').loc[self.dateindex]
        self.fret = pd.read_pickle('./data/adjopen.pkl').pct_change(fill_method=None).loc[self.dateindex]
    
    def generate_signal(self):
        # pnl = pd.concat([(self.var1.shift(2)*self.fret).mean(1),
        #                  (self.var2.shift(2)*self.fret).mean(1),
        #                  (self.var3.shift(2)*self.fret).mean(1),
        #                  (self.var4.shift(2)*self.fret).mean(1),
        #                  (self.var5.shift(2)*self.fret).mean(1),
        #                  (self.var6.shift(2)*self.fret).mean(1)], axis=1)
        # signal_mean = pd.concat([self.var1.mean(1), self.var2.mean(1), self.var3.mean(1), self.var4.mean(1), self.var5.mean(1), self.var6.mean(1)], axis=1)
        # wgt = 1/signal_mean.abs().rolling(30).mean()
        # wgt = wgt.div(wgt.sum(1), axis=0)
        # self.signal_df = self.var1.mul(wgt[0], axis=0)+ \
        #         self.var2.mul(wgt[1], axis=0)+ \
        #         self.var3.mul(wgt[2], axis=0)+\
        #         self.var4.mul(wgt[3], axis=0)+\
        #         self.var5.mul(wgt[4], axis=0)+\
        #         self.var6.mul(wgt[5], axis=0)
        self.signal_df = (self.var1+self.var2+self.var3+self.var4+self.var5+self.var6)/6
        self.beta = self.signal_df