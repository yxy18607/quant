import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing2_v1(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        high = df['high'].values; low = df['low'].values; open = df['open'].values; close = df['close'].values; returns = df['low'].pct_change(); cret = df['pct_change']
        mid = 0.5*(df['high']+df['low'])
        pskew = df[['high', 'low', 'close', 'open']].skew(1).values
        cskew = mid.rolling(5).skew().values
        ln10 = returns.rolling(10).apply(lambda x: np.sign(x).sum()).values
        cn10 = cret.rolling(10).apply(lambda x: np.sign(x).sum()).values
        cret = cret.values
        signal = np.full(len(df), 0)
        for i in range(1, len(df)):
            if ln10[i]>0:
                if pskew[i]>0:
                    signal[i] = 1
            elif ln10[i] == 0:
                if cskew[i]<0:
                    signal[i] = -1
                elif high[i]/high[i-1]-1>0:
                    signal[i] = 1
                # else:
                #     signal[i] = -1
            # else:
            #     if np.sum(np.sign(cret[i-4:i+1]))>0:
            #         signal[i] = 1
            #     else:
            #         signal[i] = -1
        return signal