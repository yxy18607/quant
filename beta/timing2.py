"""timing
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

cfg = {'startdate': '20170101',
       'enddate': '20250131',
       'instruments': ['zzqz', 'zz500', 'zz1000']}

class timing2(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        window = 5
        high = df['high'].values; low = df['low'].values; open = df['open'].values; close = df['close'].values; returns = df['pct_change'].values; volume = df['volume'].values
        signal = np.full(len(df), 0)
        for i in range(window-1, len(df)):
            if np.sum(np.sign(returns[i-window+1:i+1]))>0:
                if np.sum(returns[i-window+1:i+1])<0:
                    signal[i] = 1
                elif (high[i]-close[i]) / (close[i]-low[i]+1e-5)>=1:
                    signal[i] = 1
                else:
                    signal[i] = -1
            else:
                if np.sum(returns[i-window+1:i+1])>=0:
                    signal[i] = 1
                elif (high[i]-close[i])/(close[i]-low[i]+1e-5)>=1:
                    signal[i] = 1
                else:
                    signal[i] = -1
        return signal

    
beta = timing2(cfg)
beta()