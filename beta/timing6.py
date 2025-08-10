import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing6(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        open = df['open'].values; close = df['close'].values; high = df['high'].values; low = df['low'].values
        # f4 = df['pct_change']
        signal = np.full(len(df), 0)
        for i in range(1, len(df)):
            trd_max = np.maximum(close[i], open[i]); trd_min = np.minimum(close[i], open[i])
            exp_u = high[i]-trd_max; exp_d = trd_min-low[i]
            if exp_u>exp_d:
                if open[i]<close[i-1]:
                    signal[i] = 1
                elif close[i]<open[i]:
                    signal[i] = 1
            else:
                if open[i]>close[i-1]:
                    signal[i] = -1
                elif high[i]<high[i-1]:
                    signal[i] = -1
        return signal