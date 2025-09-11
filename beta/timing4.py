import numpy as np
import pandas as pd
import sys
import FastRolling as fr

sys.path.append('.')
from BetaModules import Beta

class timing4(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        open = df['open'].values; close = df['close'].values; high = df['high'].values; low = df['low'].values; volume = df['volume'].values
        ret = df['low'].pct_change().values
        signal = np.full(len(df), 0)
        for i in range(9, len(df)):
            x1 = np.sum(ret[i-9:i+1]); x2 = np.sum(np.sign(ret[i-9:i+1]))
            if x1>0 and x2>0:
                signal[i] = 1
            else:
                if high[i]<high[i-1]:
                    signal[i] = -1
                elif volume[i]>volume[i-1]:
                    signal[i] = 1
                else:
                    signal[i] = -1
        return signal