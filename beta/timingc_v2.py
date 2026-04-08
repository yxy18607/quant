import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timingc_v2(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def process(self):
        factor1 = pd.read_pickle('./dump/timing1.pkl')
        factor2 = pd.read_pickle('./dump/timing2.pkl')
        factor3 = pd.read_pickle('./dump/timing3.pkl')
        factor4 = pd.read_pickle('./dump/timing4.pkl')
        factor5 = pd.read_pickle('./dump/timing5.pkl')
        factor6 = pd.read_pickle('./dump/timing6.pkl')
        beta = (factor1 + factor2 + factor3 + factor4 + factor5 + factor6)/6
        self.signal_df = beta.apply(lambda col: pd.cut(col, bins=[-1, -0.6, -0.2, 0.2, 0.6, 1], labels=[-1, -0.5, 0, 0.5, 1], include_lowest=True)).astype(float)