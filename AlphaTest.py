import numpy as np
import pandas as pd
from AlphaModules import BackTest

cfg = {'startdate': '20200101',
       'enddate': '20230228',
       'alpha_id': 'alpha3_v1',
       'EmaDecay': 5,
       # 'RiskNeut': {'factors': 'size'},
       # 'NLRiskNeut': {'factors': 'ret20', 'group': 2},
      #  'IndNeut': {},
       # 'ProfitAna': {'retdays': 10, 'group': 3},
       # 'combo': False,
       # 'FactorAna': {'factors': 'size|liq|beta|mom|ret20|resvol'},
       # 'GroupFactorAna': {'factors': 'ret20|resvol'}
       }

backtest = BackTest(cfg)
backtest.exec_op()
backtest.get_position()
# backtest.exec_ana()
# backtest.dump()
# backtest()
backtest.sigbt(weight_mode=0)