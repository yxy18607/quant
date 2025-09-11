import numpy as np
import pandas as pd
import sys
import FastRolling as fr
import polars as pl
from datetime import time

sys.path.append('.')
from BetaModules import Beta

class timing7(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        open = df['open'].values; close = df['close'].values; high = df['high'].values; low = df['low'].values
        openr = df['openr'].values
        signal = np.full(len(df), 0)
        for i in range(len(df)):
            if openr[i]>0:
                signal[i] = 1
            else:
                signal[i] = -1
        return signal
    
    def pf_to_daily(self, pf: pl.LazyFrame):
        pf = (pf
              .with_columns([pl.col('time_id').dt.time().alias('time'),
                             (pl.col('close')/pl.col('pre')-1).alias('ret')])
              .group_by('date_id').agg(pl.when(pl.col('time')<=time(10, 0)).then(pl.col('ret')).otherwise(None).sum().alias('openr'))
              )
        return pf