import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing3(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        window = 3
        r1 = ((df['high']-df['open'])/(df['high']-df['low'])).rolling(window).mean().values
        r2 = ((df['close']-df['low'])/(df['high']-df['low'])).rolling(window).mean().values
        high = df['high'].values; low = df['low'].values; open = df['open'].values; close = df['close'].values; returns = df['pct_change'].values; volume = df['volume'].values
        signal = np.full(len(df), 0)
        for i in range(window-1, len(df)):
            if r1[i] >= 0.5:
                signal[i] = 1
            elif r2[i] >= 0.5:
                signal[i] = -1
            elif high[i]+low[i]-open[i]-close[i]>=0:
                signal[i] = 1
            else:
                signal[i] = -1
        return signal