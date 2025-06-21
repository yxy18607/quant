import numpy as np
import pandas as pd
import importlib
import os

# alpha_list = [f[:-3] for f in os.listdir('./alpha') if f.startswith('alpha') and f.endswith('.py')]
alpha_list = ['alpha1', 'alpha2', 'alpha3', 'alphac']
# alpha_list = ['alphac']

# mode:0——重写，1——更新
mode = 0

cfg = {'startdate': '20180101',
       'enddate': '20250516',
       }

for alpha_file in alpha_list:
    module_path = f"alpha.{alpha_file}"  # 转换为可import的路径
    module = importlib.import_module(module_path)
    if hasattr(module, alpha_file):
        alpha_class = getattr(module, alpha_file)
        alpha = alpha_class(cfg)
        alpha.generate_alpha()
        if mode:
            old = pd.read_pickle(f'./dump/{alpha_file}_v1.pkl')
            new = alpha.alpha_df.copy()
            new_index = new.index.difference(old.index)
            if len(new_index) > 0:
                new = new.loc[new_index]
                alpha.alpha_df = pd.concat([old, new], axis=0, join='outer')
                alpha.dump()
        else:
            alpha.dump()