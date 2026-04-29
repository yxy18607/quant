import numpy as np
import pandas as pd
from BetaModules import DailyCTA

inst_list = ['hs300', 'zz500', 'zz1000']
cfg = {'startdate': '20240506',
       'enddate': '20260331',
       'signal_id': 'timing6',
       # 'zone': 'hk',
       'trade_price': 'open',
       'slippage': 1,
       'fee': 0.000023,
       'instruments': [inst_list[2]],
       # 'instruments': {'zz1000': 'hstech'},
       'mode': 0,
       # 'pnl_period': 'H',
       'dump_pnl': False
       # 'update': True
       }

# pd.set_option('display.max_rows', None)
backtest = DailyCTA(**cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve(os_startdate='20240401')
backtest.plot_curve()