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
        
    def generate_signal(self, df):
        signal = np.full(len(df), 0)
        closerev = df['closeret'].values
        profit_ratio = df['closeret'].abs()/df['pct_change'].abs()
        profit_rank = fr.signal_uniform(profit_ratio, 5).values
        fear_rank = fr.signal_uniform(df['closevol'], 5).values
        for i in range(4, len(df)):
            if closerev[i]>0 and profit_rank[i]>=0:
                signal[i] = -1
            elif closerev[i]<0 and fear_rank[i]>=0:
                signal[i] = 1
        return signal
    
    def intra_to_daily(self, pf: pl.LazyFrame):
        pf = (pf
              .with_columns([pl.col('time_id').dt.time().alias('time'),
                             (pl.col('close')/pl.col('pre')-1).alias('ret')])
              .group_by('date_id').agg([pl.col('ret').filter(pl.col('time')>=time(14, 30)).sum().alias('closeret'),
                                        pl.col('volume').filter(pl.col('time')>=time(14, 30)).sum().alias('closevol')])
)
        return pf