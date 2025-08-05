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
        self.load_num = 20
        self.load_agg = True
    
    def calculate_factor(self, rlf):
        return (rlf
                .group_by('code').agg(-(pl.col('close')/pl.col('pre')-1).mean().alias('factor'))
                )
    
    def factor_to_position(self):
        return fr.pl_minmax_scalar(self.factor, 20)
        # return self.factor.with_columns(pl.col('factor').alias('pos'))