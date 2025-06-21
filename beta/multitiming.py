import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from MultiBetaModules import Beta
import FastRolling as fr
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
class multitiming(Beta):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.var1 = pd.read_pickle('./data/volS1a.pkl').loc[self.dateindex]
        self.var2 = pd.read_pickle('./data/volS2a.pkl').loc[self.dateindex]
        self.var3 = pd.read_pickle('./data/volS1.pkl').loc[self.dateindex]
        self.var4 = pd.read_pickle('./data/volS2.pkl').loc[self.dateindex]
    
    def generate_signal(self):
        window = 5
        self.beta = -(((self.var1-self.var2)/(self.var1+self.var2)-(self.var3-self.var4)/(self.var3+self.var4))**2).rolling(window).mean()
        self.signal_df = fr.signal_minmax_scaler(self.beta, 20)