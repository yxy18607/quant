import numpy as np
import pandas as pd

def ols_to_t(y: pd.DataFrame, window: int, coef_id: int):
    # 计算任何变量y关于滚动窗口时间的一阶+二阶二元回归系数: y = b0 + b1*t + b2*t^2
    # coef_id=1, 返回一阶系数; coef_id=2, 返回二阶系数
    Y = y.ffill().fillna(0).values # y的所有nan均采用前向填充，无法填充的置0
    X = np.vstack([np.ones(window), np.arange(1, window+1), np.arange(1, window+1)**2]).T # (window, 3)
    Y = np.lib.stride_tricks.sliding_window_view(Y, window_shape=(window,), axis=0) # (T-window+1, M, window)
    XTX_inv = np.linalg.pinv(X.T@X)
    X_pinv = XTX_inv @ X.T
    b = np.einsum('n,tmn->tm', X_pinv[coef_id], Y)
    beta = np.full_like(y, np.nan)
    beta[window-1:] = b
    return pd.DataFrame(beta, index=y.index, columns=y.columns)

def signal_minmax_scaler(y: pd.DataFrame, window: int):
    result = (y-y.rolling(window).min())/(y.rolling(window).max()-y.rolling(window).min())
    return result*2-1

def signal_normalize(y: pd.DataFrame, window: int, clip_level: float=1):
    result = (y - y.rolling(window).mean()) / y.rolling(window).std()
    return result.clip(-clip_level, clip_level)

def signal_uniform(y: pd.DataFrame, window: int):
    return y.rolling(window).rank(pct=True)*2-1