import numpy as np
import pandas as pd
import polars as pl

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

def pl_minmax_scalar(lf: pl.DataFrame, window: int, src: str='factor', dst: str='pos'):
    return (lf.with_columns([pl.col(src).rolling_max(window).over('code').alias('fmax'),
                             pl.col(src).rolling_min(window).over('code').alias('fmin')])
            .with_columns(((pl.col(src)-pl.col('fmin'))/(pl.col('fmax')-pl.col('fmin'))*2-1).alias(dst)))

def pl_pct_rank(lf: pl.DataFrame, window: int, src: str='factor', dst: str='pos'):
    df = lf.to_pandas()
    df[dst] = df.groupby('code')[src].transform(lambda col: col.rolling(window).rank(pct=True)*2-1)
    return pl.from_pandas(df)

def pl_pos_discretize(lf: pl.DataFrame, cut: int=5):
    df = lf.to_pandas()
    df['pos'] = pd.cut(df['pos'], bins=np.linspace(-1, 1, 2*cut+2), labels=np.linspace(-1, 1, 2*cut+1), include_lowest=True).astype(float)
    return pl.from_pandas(df)

def pl_normalizer(lf: pl.DataFrame, window: int, src: str='factor', dst: str='pos'):
    return (lf.with_columns([pl.col(src).rolling_mean(window).over('code').alias('favg'),
                             pl.col(src).rolling_std(window).over('code').alias('fstd')])
            .with_columns(((pl.col(src)-pl.col('favg'))/pl.col('fstd')).clip(-1, 1).alias(dst)))