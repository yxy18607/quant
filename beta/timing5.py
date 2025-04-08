"""timing
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')
from BetaModules import Beta

class timing5(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def generate_signal(self):
        factor1 = pd.read_pickle('./dump/timing1.pkl')
        factor2 = pd.read_pickle('./dump/timing2.pkl')
        self.signal_df = (factor1+factor2)/2