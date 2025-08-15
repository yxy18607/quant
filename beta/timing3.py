import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing3(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        dp = df['pct_change'].values; close = df['close'].values; low = df['low'].values; open = df['open'].values; high = df['high'].values
        f1 = df['pct_change'].rolling(5).sum()/df['pct_change'].abs().rolling(5).sum()
        s1 = f1>0.5; s2 = f1<-0.5; s3 = (f1<=0.5)&(f1>0); s4 = (f1<=0)&(f1>=-0.5)
        f2 = (df['open']-df['low'])/(df['high']-df['low'])
        f3 = (df['close']-df['low'])
        f3 = f3-f3.rolling(3).mean()
        signal = np.full(len(df), 0)
        for i in range(4, len(df)):
            trend = np.sum(dp[i-4:i+1])/np.sum(np.abs(dp[i-4:i+1]))
            if trend>0.5:
                # open_power = (open[i]-low[i])/(high[i]-low[i])
                # if open_power>0.5:
                if open[i]>close[i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 1
            elif trend<=0.5 and trend>0:
                # midd = close[i]+open[i]-close[i-1]-open[i-1]
                # if midd>0:
                if close[i]-low[i]-np.mean(close[i-2:i+1]-low[i-2:i+1])>0:
                    signal[i] = -1
                else:
                    signal[i] = 1
            # elif trend<-0.5:
            #     if close[i]-low[i]-np.mean(close[i-2:i+1]-low[i-2:i+1])>0:
            #         signal[i] = -1
            #     else:
            #         signal[i] = 1

        return signal