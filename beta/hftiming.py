import numpy as np
import pandas as pd
import sys
import polars as pl
import FastRolling as fr

sys.path.append('..')
from MultiBetaModules import Beta

class hftiming(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def generate_daily(self, df_lazy):
        return df_lazy.with_columns(
            (pl.col('close').cast(pl.Float64)/pl.col('pre').cast(pl.Float64)-1).alias('ret')).group_by('code').agg(
            ((pl.col('ret')-pl.col('ret').mean())/pl.col('ret').std()).abs().gt(1).mean().alias('factor'))
    
    def generate_signal(self):
        beta = self.daily_df.rolling(5).mean()
        signal = fr.signal_minmax_scaler(beta, 20)
        self.signal_df = signal