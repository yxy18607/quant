import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing1(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        open = df['open'].values; high = df['high'].values; low = df['low'].values; close = df['close'].values; volume = df['volume'].values; ret = df['pct_change']
        signal = np.full(len(df), 0)
        for i in range(1, len(df)):
            if open[i] > 0.5*(high[i]+low[i]) and close[i] > 0.5*(high[i]+low[i]):
                signal[i] = -1
            elif open[i] < 0.5*(high[i]+low[i]) and close[i] > 0.5*(high[i]+low[i]):
                if volume[i]>volume[i-1]:
                    signal[i] = 1
                else:
                    signal[i] = -1
            elif open[i] < 0.5*(high[i]+low[i]) and close[i] < 0.5*(high[i]+low[i]):
                signal[i] = 1
        return signal