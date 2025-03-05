"""timing
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

cfg = {'startdate': '20170101',
       'enddate': '20250228',
       'instruments': ['hs300', 'zz500', 'zz1000']}

class timing1_v1(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        open = df['open'].values; high = df['high'].values; low = df['low'].values; close = df['close'].values; volume = df['volume'].values
        signal = np.full(len(df), 0)
        for i in range(1, len(df)):
            if open[i] > 0.5*(high[i]+low[i]) and close[i] > 0.5*(high[i]+low[i]):
                signal[i] = -1
            elif open[i] < 0.5*(high[i]+low[i]) and close[i] > 0.5*(high[i]+low[i]):
                if open[i-1] < 0.5*(high[i-1]+low[i-1]) and close[i-1] > 0.5*(high[i-1]+low[i-1]):
                    signal[i] = 1
            elif open[i] < 0.5*(high[i]+low[i]) and close[i] < 0.5*(high[i]+low[i]):
                signal[i] = 1
        return signal

beta = timing1_v1(cfg)
beta()