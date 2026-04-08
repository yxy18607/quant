import numpy as np
import pandas as pd
import sys
import FastRolling as fr
import polars as pl
from datetime import time

sys.path.append('.')
from BetaModules import Beta

class timing6(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_signal(self, df):
        open = df['open'].values; close = df['close'].values; high = df['high'].values; low = df['low'].values; vol = df['volume'].values
        signal = np.full(len(df), 0)
        skd = df['skew'].values
        for i in range(2, len(df)):
            if skd[i]>skd[i-1] and high[i]>high[i-1]:
                signal[i] = 1
            elif skd[i]<skd[i-1] and high[i]<high[i-1]:
                signal[i] = -1
        return signal
    
    def intra_to_daily(self, pf: pl.LazyFrame):
        pf = (pf
              .with_columns([pl.col('time_id').dt.time().alias('time'),
                             (pl.col('close')/pl.col('pre')-1).alias('ret')])
              .group_by('date_id').agg([pl.col('close').skew().alias('skew'),
              ])
)
        return pf