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
        return rlf.group_by('code').agg(pl.col('close').std().alias('factor'))
    
    def factor_to_position(self):
        return (self.factor-self.factor.rolling_min(20))/(self.factor.rolling_max(20)-self.factor.rolling_min(20))*2-1