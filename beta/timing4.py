import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing4(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        window = 6
        ret = (df['high']+df['low']-(df['close']+df['open'])).values; volume = df['volume'].values
        signal = np.full(len(df), 0)
        for i in range(window-1, len(df)):
            subret = ret[i-window+1:i+1]
            upret = subret[subret>0].mean() if len(subret[subret>0])>0 else 1e-4
            dnret = np.abs(subret[subret<0]).mean() if len(subret[subret<0])>0 else 1e-4
            if ret[i]>=0 and upret/dnret>=1:
                signal[i] = 1
            else:
                if ret[i-1:i+1].sum()>0 and volume[i]>volume[i-1]:
                    signal[i] = 1
                elif ret[i-1:i+1].sum()<0 and volume[i]<volume[i-1]:
                    signal[i] = -1
        return signal