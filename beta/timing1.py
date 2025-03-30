"""timing
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

cfg = {'startdate': '20200101',
       'enddate': '20250228',
       'instruments': ['300779']}

class timing1(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_beta(self, df):
        open = df['open'].values; high = df['high'].values; low = df['low'].values; close = df['close'].values
        signal = np.full(len(df), 0)
        signal[(open>=0.5*(high+low))&(close>=0.5*(high+low))] = -1
        signal[(open<0.5*(high+low))&(close<0.5*(high+low))] = 1
        return signal

beta = timing1(cfg)
beta()