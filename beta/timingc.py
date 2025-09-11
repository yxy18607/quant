import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timingc(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_signal(self):
        factor1 = pd.read_pickle('./dump/timing1.pkl')
        factor2 = pd.read_pickle('./dump/timing2.pkl')
        factor3 = pd.read_pickle('./dump/timing3.pkl')
        beta = (factor1+factor2+factor3)/3
        self.signal_df = beta#.apply(lambda col: pd.cut(col, bins=[-1, -0.3, 0.3, 1], labels=[-1, 0, 1], include_lowest=True)).astype(float)