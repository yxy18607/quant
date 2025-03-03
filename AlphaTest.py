import numpy as np
import pandas as pd
from AlphaModules import BackTest

cfg = {'startdate': '20200101',
       'enddate': '20250228',
       'alpha_id': 'alpha2_v1',
       'EmaDecay': 12,
      #  'RiskNeut': {'factors': 'resvol|ret20'},
      #  'NLRiskNeut': {'factors': 'resvol|ret20', 'group': 4},
      #  'IndNeut': {},
       'ProfitAna': {'retdays': 10, 'group': 3},
       'combo': False,
      #  'FactorAna': {'factors': 'size|liq|beta|mom|ret20|resvol'},
      #  'GroupFactorAna': {'factors': 'resvol|ret20'}
       }

backtest = BackTest(cfg)
# backtest.exec_op()
# backtest.get_position()
# backtest.exec_ana()
# backtest.dump()
backtest()
# backtest.sigbt(300)