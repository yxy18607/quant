import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timingc_v1(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def process(self):
        factor2 = pd.read_pickle('./dump/timing2.pkl')
        factor3 = pd.read_pickle('./dump/timing3.pkl')
        factor4 = pd.read_pickle('./dump/timing4.pkl')
        factor5 = pd.read_pickle('./dump/timing5.pkl')
        beta = (factor2+factor3+factor4+factor5)/4
        self.signal_df = beta.apply(lambda col: pd.cut(col, bins=[-1, -0.5, -0.25, 0.2, 0.4, 1], labels=[-1, -0.5, 0, 0.5, 1], include_lowest=True)).astype(float)