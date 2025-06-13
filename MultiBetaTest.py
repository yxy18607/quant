import numpy as np
import pandas as pd
from MultiBetaModules import BackTest

cfg = {'startdate': '20180101',
       'enddate': '20230531',
       # 'signal_id': 'multitiming',
       # 'signal_id': 'hftiming',
       'signal_id': 'stocklgbm',
       # 'signal_id': 'stocktiming6',
       'mode': 0,
       'fee': 0.0000
       }

backtest = BackTest(cfg)
backtest()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve()
# backtest.plot_curve(os_startdate='20230101')
# backtest.save_pnl()