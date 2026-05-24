import numpy as np
import pandas as pd
from BetaModules import DailyCTA

inst_list = ['hs300', 'zz500', 'zz1000']

# ============================================================
# cfg —— 纯 DailyCTA 配置（标量形式）
# ============================================================
cfg = {'startdate': '20180101',
       'enddate': '20260430',
       'signal_id': 'timingc_v3',
       'instruments': [inst_list[2]],
       # 'instruments': {'zz1000': 'hstech'},
       # 'zone': 'hk',
       'trade_price': 'open',
       'slippage': 1,
       'fee': 0.000023,
       'mode': 0,
       # 'pnl_period': 'H',
       # 'dump_pnl': False,
       # 'update': True,
       }

# pd.set_option('display.max_rows', None)

# ============================================================
# 循环参数 —— 三者互斥，至多一个为非 None
#   loop_signals:   多信号逐一遍历      例 ['timing1', 'timing2']
#   loop_instruments: 逐品种遍历（每次单品种传入）  例 ['zz1000', 'zz500', 'hs300']
#   loop_dates:     多时段逐一遍历      例 [('20180101','20201231'), ('20210101','20260430')]
# ============================================================
loop_signals = None
# loop_signals = ['timing1', 'timing2', 'timing3']
loop_instruments = None
# loop_instruments = ['hs300', 'zz500', 'zz1000']
loop_dates = None
# loop_dates = [('20180101', '20240430'), ('20240501', '20260430')]


# ---------- 以下为执行逻辑，无需改动 ----------

def _count_active(*args):
    return sum(1 for x in args if x is not None)

_active = _count_active(loop_signals, loop_instruments, loop_dates)
if _active > 1:
    raise ValueError(f"loop_signals / loop_instruments / loop_dates 至多一个有效，当前 {_active} 个均非 None")

_batch = []

if loop_signals is not None:
    for sid in loop_signals:
        cfg['signal_id'] = sid
        _batch.append(dict(cfg))

elif loop_instruments is not None:
    for inst in loop_instruments:
        c = dict(cfg)
        c['instruments'] = [inst]
        _batch.append(c)

elif loop_dates is not None:
    for sd, ed in loop_dates:
        c = dict(cfg)
        c['startdate'] = sd
        c['enddate'] = ed
        _batch.append(c)

else:
    _batch.append(dict(cfg))

for c in _batch:
    backtest = DailyCTA(**c)
    backtest()
    # backtest.plot_curve()
