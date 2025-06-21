import numpy as np
import pandas as pd
from AlphaModules import BackTest

cfg = {'startdate': '20200103',
       'enddate': '20250331',
       'alpha_id': 'alphac_v1',
       'EmaDecay': 10,
       # 'tvr_day': ['01', '16']
       'RiskNeut': {'factors': 'resvol'},
       # 'NLRiskNeut': {'factors': 'ret20', 'group': 2},
      #  'IndNeut': {},
       # 'ProfitAna': {'retdays': 5, 'group': 10},
       # 'combo': False,
       'FactorAna': {'factors': 'size|liq|beta|mom|ret20|resvol'},
       # 'GroupFactorAna': {'factors': 'ret20|resvol'}
       }

backtest = BackTest(cfg)
# backtest.exec_op()
# backtest.get_position()
# backtest.exec_ana()
backtest()
# backtest.dump()
backtest.sigbt(topN=500, weight_mode=1, benchmark='zzqz')

# timing_ts = pd.read_csv('./data_attached/大盘择时未来10日信号_涨1跌0震荡2.csv', header=None, index_col=0).squeeze()
# timing_ts.index = timing_ts.index.astype('str')
# timing_ts = timing_ts.map({1: 1, 0: 0.5, 2: 0.8})
# backtest.get_position()
# backtest.sigbt(topN=200, weight_mode=0, benchmark='zzqz', timing_ts=timing_ts)