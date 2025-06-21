import numpy as np
import pandas as pd
from HighFreqModules import BackTest

cfg = {'startdate': '20180101',
       'enddate': '20181231',
       'signal_id': 'hfetf',
       # 'signal_id': 'stocktiming6',
       'fee': 0,
       'load_unit': '60m',
       'load_num': 10
       }

backtest = BackTest(cfg)
backtest.test()
# backtest.get_pnl()
# backtest.profit_ana()
# backtest.plot_curve()
# backtest.plot_curve(os_startdate='20230601')
# backtest.save_pnl()