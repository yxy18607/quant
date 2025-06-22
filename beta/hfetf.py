import numpy as np
import pandas as pd
import sys
import polars as pl
import FastRolling as fr

sys.path.append('..')
from HighFreqModules import BackTest

class hfetf(BackTest):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def calculate_factor(self, rlf):
        return rlf.group_by('code').agg(pl.col('close').cast(pl.Float64).std().alias('factor'))
    
    def factor_to_position(self):
        return self.factor.with_columns([pl.col('factor').over('code').rolling_max(20).alias('fmax'),
                                         pl.col('factor').over('code').rolling_min(20).alias('fmin')]).with_columns(
                                         ((pl.col('factor')-pl.col('fmin'))/(pl.col('fmax')-pl.col('fmin'))).alias('pos')).select(
                                             [pl.col('code'), pl.col('datetime'), pl.col('factor'), pl.col('pos')]
                                         )