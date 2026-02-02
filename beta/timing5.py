import numpy as np
import pandas as pd
import sys
import FastRolling as fr
import polars as pl
from datetime import time

sys.path.append('.')
from BetaModules import Beta

class timing5(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_signal(self, df):
        open = df['open'].values; close = df['close'].values; high = df['high'].values; low = df['low'].values; vol = df['volume'].values
        signal = np.full(len(df), 0)
        window =10 
        rs = df['pct_change'].rolling(window).std().values
        vol = fr.sma(df['volume'], window).values
        sk = df['skew'].values
        exr = df['extup'].values
        for i in range(10, len(df)):
            if rs[i]>rs[i-1] and vol[i]>vol[i-1]:
                signal[i] = 1
            elif rs[i]<rs[i-1] and sk[i]<0:
                signal[i] = -1
            else:
                if exr[i]>exr[i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 1
        return signal
    
    def intra_to_daily(self, pf: pl.LazyFrame):
        pf = (pf
              .with_columns([pl.col('time_id').dt.time().alias('time'),
                             (pl.col('close')/pl.col('pre')-1).alias('ret')])
              .group_by('date_id').agg([pl.col('close').skew().alias('skew'),
                                        pl.col('ret').filter(pl.col('ret')>pl.col('ret').mean()+pl.col('ret').std()).sum().alias('extup')
])
)
        return pf