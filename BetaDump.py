import numpy as np
import pandas as pd
import importlib
import os

# beta_list = [f[:-3] for f in os.listdir('./beta') if f.startswith('beta') and f.endswith('.py')]
beta_list = ['hktiming']
# beta_list = ['timing1', 'timing2', 'timing5']

# mode:0——重写，1——更新
mode = 0

cfg = {'startdate': '20180101',
       'enddate': '20250530',
    #    'instruments': ['zz1000', 'zz500', 'hs300', 'sz50'],
       'instruments': ['hsi'],
       'zone': 'hk'
}

for beta_file in beta_list:
    module_path = f"beta.{beta_file}"  # 转换为可import的路径
    module = importlib.import_module(module_path)
    if hasattr(module, beta_file):
        beta_class = getattr(module, beta_file)
        beta = beta_class(cfg)
        beta.generate_signal()
        if mode:
            old = pd.read_pickle(f'./dump/{beta_file}.pkl')
            new = beta.signal_df.copy()
            new_index = new.index.difference(old.index)
            if len(new_index) > 0:
                new = new.loc[new_index]
                beta.signal_df = pd.concat([old, new], axis=0, join='outer')
                beta.dump()
        else:
            beta.dump()