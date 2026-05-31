import numpy as np
import pandas as pd
from MultiBetaModules import BackTest

# performance = pd.read_pickle('./pnl/etftiming1.stats_cross.pkl')
# instruments = performance.loc[(performance['sample']>=0.8)&(performance['dateitv'].str[-8:]=='20231130'), 'annsp'].sort_values(ascending=False).iloc[:100].index.tolist()

cfg = {'startdate': '20240101',
       'enddate': '20251231',
       # 'signal_id': 'stocktiming1',
       'signal_id': 'stockplus',
       # 'signal_id': 'etftiming1',
       'category': 'stock',
       # 'instruments': instruments,
       # 'signal_id': 'stocktiming6',
       'mode': 0,
       'fee': 0.00005,
       # 'dump_pnl': False
       }

backtest = BackTest(**cfg)
backtest()
# backtest.plot_curve()
# backtest.plot_curve(os_startdate='20231031')
# backtest.stats_cross()