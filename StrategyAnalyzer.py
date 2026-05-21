"""
策略绩效分析与调试工具集 (Strategy Performance Analysis Toolkit)

提供量化策略分析常用功能：
  - 条件收益率统计 (structure_stats / annual_stat)
  - 策略绩效回顾 (performance_review)
  - 分层测试 / Decile Analysis (decile_analysis)
  - 滚动分布监控 (rolling_stats)
  - 离散仓位单调性分析 (position_analysis)
  - 仓位暴露度分析 (exposure_analysis)
  - 策略间相关性分析 (correlation_analysis)

使用方式：
  from StrategyAnalyzer import StrategyAnalyzer, structure_stats, annual_stat

  # 快速使用独立函数
  structure_stats(fret, cond1, cond2, cond3)

  # 或实例化分析器（可自定义数据目录）
  sa = StrategyAnalyzer()
  sa.performance_review('timing1', '20180101', '20260331')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Optional, Union, Tuple


# ============================================================================
# Internal helpers
# ============================================================================

def _dd_duration(nav: pd.Series) -> int:
    """计算最长回撤持续天数（峰值到恢复的交易日数）。"""
    diff = nav.cummax().diff()
    # 首日 diff 为 NaN，视为新峰值起始
    groups = diff.fillna(1).astype(bool).cumsum()
    counts = np.bincount(groups)
    return counts.max() if len(counts) > 0 else 0


def _plot_decile_bars(ax, labels, values, title, color):
    """在给定轴上绘制 decile 柱状图并标注数值。"""
    vals = np.asarray(values).ravel()
    bars = ax.bar(range(len(labels)), vals, color=color, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    for bar in bars:
        yval = bar.get_height()
        offset = 0.015 * yrange
        va = 'bottom' if yval >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2,
                yval + offset if yval >= 0 else yval - offset,
                f'{yval:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')


def _fmt_print(df: pd.DataFrame, fmt: dict):
    """按指定格式打印 DataFrame（不修改原数据）。"""
    out = df.copy()
    for col, f in fmt.items():
        if col not in out.columns:
            continue
        if f == '%':
            out[col] = out[col].map(lambda x: f'{x:.2%}' if pd.notna(x) else '')
        else:
            out[col] = out[col].map(lambda x: f'{x:{f}}' if pd.notna(x) else '')
    pd.set_option('display.expand_frame_repr', False)
    print(out)


# ============================================================================
# StrategyAnalyzer Class
# ============================================================================

class StrategyAnalyzer:
    """
    策略绩效分析工具类。

    Parameters
    ----------
    pnl_dir : str
        PnL 文件目录（默认 ./pnl）。
    dump_dir : str
        信号 dump 文件目录（默认 ./dump）。
    data_dir : str
        市场数据文件目录（默认 ./data）。
    """

    def __init__(self, pnl_dir: str = './pnl', dump_dir: str = './dump', data_dir: str = './data'):
        self.pnl_dir = pnl_dir
        self.dump_dir = dump_dir
        self.data_dir = data_dir

    # ========================================================================
    # Static Methods — 纯计算，不依赖文件读写
    # ========================================================================

    @staticmethod
    def structure_stats(fret: pd.Series, *states: pd.Series, verbose: bool = True) -> pd.DataFrame:
        """
        条件收益率统计：在不同市场状态下统计未来收益率的分布特征。

        Parameters
        ----------
        fret : pd.Series
            未来收益率序列（日频）。
        *states : pd.Series (bool)
            一组条件布尔掩码，每个与 fret 同 index。
        verbose : bool
            是否打印格式化结果。

        Returns
        -------
        pd.DataFrame
            列：count, wgt, annret, annstd, winr, odd, kelly, annsp。
        """
        ann_div = len(fret) / 250
        records = []
        for state in states:
            sf = fret[state]
            n = len(sf)
            if n == 0:
                records.append({'count': 0, 'wgt': 0, 'annret': np.nan, 'annstd': np.nan,
                                'winr': np.nan, 'odd': np.nan, 'kelly': np.nan, 'annsp': np.nan})
                continue
            annret = sf.mean() * n / ann_div
            annstd = sf.std() * np.sqrt(n / ann_div)
            if sf.mean() > 0:
                winr = (sf > 0).mean()
                odd = -sf[sf > 0].mean() / sf[sf < 0].mean() if (sf < 0).any() else np.inf
            else:
                winr = (sf < 0).mean()
                odd = -sf[sf < 0].mean() / sf[sf > 0].mean() if (sf > 0).any() else np.inf
            records.append({
                'count': n,
                'wgt': n / len(fret),
                'annret': annret,
                'annstd': annstd,
                'winr': winr,
                'odd': odd,
                'kelly': winr * odd - (1 - winr),
                'annsp': np.abs(sf.mean() / sf.std()) * np.sqrt(n / ann_div) if sf.std() > 0 else 0,
            })
        result = pd.DataFrame(records, index=np.arange(1, len(states) + 1))
        if verbose:
            _fmt_print(result, {'wgt': '%', 'annret': '%', 'annstd': '%', 'winr': '%',
                                'odd': '.4f', 'kelly': '.4f', 'annsp': '.2f'})
        return result

    @staticmethod
    def annual_stat(fret: pd.Series, state: pd.Series, verbose: bool = True) -> pd.DataFrame:
        """
        按年份拆分统计条件收益率。

        Parameters
        ----------
        fret : pd.Series
            未来收益率序列（完整样本）。
        state : pd.Series (bool)
            条件布尔掩码。
        verbose : bool
            是否打印格式化结果。

        Returns
        -------
        pd.DataFrame
            按年份索引，列：count, wgt, annret, annstd, winr, odd, kelly, annsp。
        """
        df = pd.DataFrame({'fret': fret[state]})
        df['year'] = df.index.str[:4]
        ann_num = (pd.DataFrame({'fret': fret, 'year': fret.index.str[:4]})
                   .groupby('year')['fret'].count())
        ann_div = ann_num / 250
        group = df.groupby('year')['fret']
        count = group.count()
        annret = group.mean() * count / ann_div
        annstd = group.std() * np.sqrt(count / ann_div)
        if df['fret'].mean() > 0:
            winr = group.apply(lambda x: (x > 0).mean())
            odd = group.apply(lambda x: -x[x > 0].mean() / x[x < 0].mean() if (x < 0).any() else np.inf)
        else:
            winr = group.apply(lambda x: (x < 0).mean())
            odd = group.apply(lambda x: -x[x < 0].mean() / x[x > 0].mean() if (x > 0).any() else np.inf)
        kelly = winr * odd - (1 - winr)
        annsp = annret / annstd.replace(0, np.nan)
        result = pd.DataFrame({
            'count': count,
            'wgt': count / ann_num,
            'annret': annret,
            'annstd': annstd,
            'winr': winr,
            'odd': odd,
            'kelly': kelly,
            'annsp': annsp,
        })
        if verbose:
            _fmt_print(result, {'wgt': '%', 'annret': '%', 'annstd': '%', 'winr': '%',
                                'odd': '.4f', 'kelly': '.2f', 'annsp': '.2f'})
        return result

    @staticmethod
    def annual_metric(pnl: pd.Series, coef_ann: float) -> Tuple[float, float, int]:
        """
        计算年化收益率、夏普比率和样本数。

        Parameters
        ----------
        pnl : pd.Series
            日收益率序列。
        coef_ann : float
            年化系数，通常为 len(全样本) / 250。

        Returns
        -------
        (annret, annsp, count)
        """
        n = len(pnl)
        if n == 0:
            return 0.0, 0.0, 0
        ret = pnl.mean() * n / coef_ann
        sp = pnl.mean() / pnl.std() * np.sqrt(n / coef_ann) if pnl.std() > 0 else 0.0
        return ret, sp, n

    @staticmethod
    def decile_analysis(
        indicator: pd.Series,
        pnl: pd.Series,
        n_groups: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ax: Optional[Union[plt.Axes, np.ndarray]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        分层测试（Decile Analysis）：按市场监控指标的分位数组统计策略表现。

        Parameters
        ----------
        indicator : pd.Series
            市场监控指标（如波动率、换手率等）。
        pnl : pd.Series
            策略日收益率序列。
        n_groups : int
            分组数量，默认 5。
        start_date, end_date : str, optional
            截取时间区间（YYYY-MM-DD 或 YYYYMMDD）。
        ax : matplotlib Axes, optional
            指定绘图轴（需 2 个）。若为 None 则自动创建。
        verbose : bool
            是否绘图。

        Returns
        -------
        pd.DataFrame
            各组年化收益率和夏普比率。
        """
        df = pd.concat([indicator.rename('indicator'), pnl.rename('pnl')], axis=1).dropna()
        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]
        df['group'] = pd.qcut(df['indicator'], q=n_groups,
                              labels=[f'G{i + 1}' for i in range(n_groups)],
                              duplicates='drop')
        result = pd.DataFrame({
            'annret': df.groupby('group', observed=True)['pnl'].apply(lambda x: x.mean() * 252),
            'sharpe': df.groupby('group', observed=True)['pnl'].apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0),
        })
        if verbose:
            if ax is None:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            else:
                axes = ax
            title_prefix = ''
            if start_date or end_date:
                s = start_date or df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(df.index[0])
                e = end_date or df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                title_prefix = f' ({s} to {e})'
            _plot_decile_bars(axes[0], result.index, result['annret'],
                              f'Annualized Return by Indicator Quantiles{title_prefix}', '#4C72B0')
            _plot_decile_bars(axes[1], result.index, result['sharpe'],
                              f'Annualized Sharpe by Indicator Quantiles{title_prefix}', '#C44E52')
            plt.tight_layout()
            plt.show()
        return result

    @staticmethod
    def rolling_stats(
        pnl: pd.Series,
        window: int = 120,
        start: Optional[str] = None,
        end: Optional[str] = None,
        ax: Optional[Union[plt.Axes, np.ndarray]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        PnL 滚动分布监控：滚动均值、标准差、偏度、峰度。

        Parameters
        ----------
        pnl : pd.Series
            策略日收益率序列。
        window : int
            滚动窗口大小，默认 120。
        start, end : str, optional
            截取时间区间（YYYYMMDD）。
        ax : matplotlib Axes, optional
            指定绘图轴（需 4 个）。
        verbose : bool
            是否绘图。

        Returns
        -------
        pd.DataFrame
            列：mean, std, skew, kurt。
        """
        if start:
            pnl = pnl[start:]
        if end:
            pnl = pnl[:end]
        pnl = pnl.dropna()
        df = pd.DataFrame({
            'mean': pnl.rolling(window).mean(),
            'std': pnl.rolling(window).std(),
            'skew': pnl.rolling(window).skew(),
            'kurt': pnl.rolling(window).kurt(),
        })
        if verbose:
            if ax is None:
                fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            else:
                axes = ax
            for i, col in enumerate(['mean', 'std', 'skew', 'kurt']):
                axes[i].plot(df.index, df[col], linewidth=0.8)
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
            axes[0].set_title(f'Rolling Stats (window={window})')
            plt.tight_layout()
            plt.show()
        return df

    # ========================================================================
    # Instance Methods — 基于文件读写的策略分析
    # ========================================================================

    def performance_review(
        self,
        pnl: Union[str, pd.DataFrame],
        start: str,
        end: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        策略绩效全面回顾：年化收益、夏普、最大回撤、Calmar、胜率、赔率、换手率等。
        按年度拆分，并在末尾汇总全区间。

        Parameters
        ----------
        pnl : str or pd.DataFrame
            策略 PnL 文件名（不含 .pnl.pkl 后缀）或包含 'pnl' 列的 DataFrame。
        start, end : str
            统计区间（YYYYMMDD）。
        verbose : bool
            是否打印结果。

        Returns
        -------
        pd.DataFrame
            MultiIndex (from, to) 绩效表。
        """
        if isinstance(pnl, str):
            pnl = pd.read_pickle(f'{self.pnl_dir}/{pnl}.pnl.pkl')
        pnl = pnl.query('@start <= index <= @end').copy()
        pnl.index = pd.to_datetime(pnl.index)
        dateindex = pnl.index
        pnl['nav'] = pnl['pnl'].cumsum()
        pnl['year'] = pnl.index.year.values
        has_dpos = 'dpos' in pnl.columns

        # --- 全区间 ---
        ret = pnl['pnl'].mean() * 250
        sp = pnl['pnl'].mean() / pnl['pnl'].std() * np.sqrt(250) if pnl['pnl'].std() > 0 else 0
        pnl['dd_T'] = pnl['nav'].cummax() - pnl['nav']
        mdd = pnl['dd_T'].max()
        dd_end = pnl['dd_T'].idxmax()
        dd_start = pnl.loc[:dd_end, 'nav'].idxmax()
        dd_dur = _dd_duration(pnl['nav'])
        pos = pnl['pnl'] > 0
        neg = pnl['pnl'] < 0
        n_pos, n_neg = pos.sum(), neg.sum()
        winr = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        odd = -pnl.loc[pos, 'pnl'].mean() / pnl.loc[neg, 'pnl'].mean() if n_neg > 0 else np.inf
        calmar = ret / mdd if mdd > 0 else np.inf
        tvr = pnl['dpos'].mean() * 250 if has_dpos else np.nan

        # --- 年度 ---
        gpnl = pnl.groupby('year')
        pnl['dd_y'] = gpnl['nav'].cummax() - pnl['nav']
        ret_y = gpnl['pnl'].mean() * 250
        sp_y = gpnl['pnl'].mean() / gpnl['pnl'].std() * np.sqrt(250)
        mdd_y = gpnl['dd_y'].max()
        dd_end_y = gpnl['dd_y'].idxmax()
        dd_start_y = gpnl.apply(
            lambda x: x.loc[:dd_end_y[x.name], 'nav'].idxmax()
            if pd.notna(dd_end_y.get(x.name)) else pd.NaT,
            include_groups=False,
        )
        dd_dur_y = gpnl['nav'].apply(_dd_duration)
        winr_y = gpnl['pnl'].apply(
            lambda x: ((x > 0).sum() / ((x > 0).sum() + (x < 0).sum()))
            if ((x > 0).sum() + (x < 0).sum()) > 0 else 0)
        odd_y = -gpnl['pnl'].apply(
            lambda x: x[x > 0].mean() / x[x < 0].mean() if (x < 0).sum() > 0 else np.inf)
        calmar_y = ret_y / mdd_y.replace(0, np.nan)
        tvr_y = gpnl['dpos'].mean() * 250 if has_dpos else pd.Series(np.nan, index=ret_y.index)

        # --- 组装 ---
        idx_from = gpnl.head(1).index
        idx_to = gpnl.tail(1).index
        out = pd.DataFrame({
            'annret': ret_y.values * 100,
            'annsp': sp_y.values,
            'mdd': mdd_y.values * 100,
            'mdd_start': dd_start_y.values,
            'mdd_end': dd_end_y.values,
            'mdd_dur': dd_dur_y.values,
            'anntvr': tvr_y.values,
            'winr': winr_y.values * 100,
            'odd': odd_y.values,
            'calmar': calmar_y.values,
        }, index=pd.MultiIndex.from_arrays([idx_from, idx_to], names=['from', 'to']))
        # 追加全区间汇总行
        out.loc[(dateindex[0], dateindex[-1]), :] = [
            ret * 100, sp, mdd * 100, dd_start, dd_end, dd_dur,
            tvr, winr * 100, odd, calmar,
        ]
        out = out.round(2)
        if verbose:
            pd.set_option('display.expand_frame_repr', False)
            print(out)
        return out

    def position_analysis(
        self,
        eq_name: str,
        bins: int,
        signal_col: str = 'zz1000',
        shift: int = 2,
        start: str = '20180101',
        end: str = '20260331',
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        离散仓位单调性分析：统计不同绝对仓位水平下的策略表现。

        按信号绝对值等距分为 (bins+1) 档，计算每档的年化收益、夏普和权重。

        Parameters
        ----------
        eq_name : str
            策略文件名（不含 .pnl.pkl / .pkl 后缀）。
        bins : int
            仓位离散化档位数（如 7 则分 0/7, 1/7, ..., 7/7）。
        signal_col : str
            dump 文件中的信号列名，默认 'zz1000'。
        shift : int
            信号前置天数，默认 2。
        start, end : str
            统计区间（YYYYMMDD）。
        verbose : bool
            是否打印结果。

        Returns
        -------
        pd.DataFrame
            index 为 bin 标签，列：annret, annsp, wgt。
        """
        pnl = pd.read_pickle(f'{self.pnl_dir}/{eq_name}.pnl.pkl').query('@start <= index <= @end')['pnl']
        signal = (pd.read_pickle(f'{self.dump_dir}/{eq_name}.pkl')
                  .query('@start <= index <= @end')[signal_col]
                  .shift(shift))
        coef_ann = len(pnl) / 250
        records = []
        for i in range(0, bins + 1):
            r, s, n = self.annual_metric(pnl[signal.abs() == i / bins], coef_ann)
            records.append({'bin': f'{i}/{bins}', 'annret': r, 'annsp': s, 'wgt': n / len(pnl)})
        result = pd.DataFrame(records).set_index('bin')
        if verbose:
            for _, row in result.iterrows():
                print(f"abs(pos)--{row.name}: annret {row['annret']:.2%}, "
                      f"sp {row['annsp']:.2f}, len {row['wgt']:.2%}")
        return result

    def exposure_analysis(
        self,
        port_sig_name: str,
        sub_sig_list: List[str],
        signal_col: str = 'zz1000',
        start: str = '20180101',
        end: str = '20260331',
        verbose: bool = True,
    ) -> pd.Series:
        """
        仓位暴露度分析：计算子策略信号在组合信号上的暴露度（内积均值）。

        Parameters
        ----------
        port_sig_name : str
            组合策略文件名（不含 .pkl 后缀）。
        sub_sig_list : list of str
            子策略文件名列表。
        signal_col : str
            dump 文件中的信号列名，默认 'zz1000'。
        start, end : str
            统计区间（YYYYMMDD）。
        verbose : bool
            是否打印结果。

        Returns
        -------
        pd.Series
            各子策略对组合的暴露度。
        """
        port_sig = (pd.read_pickle(f'{self.dump_dir}/{port_sig_name}.pkl')
                    .query('@start <= index <= @end')[signal_col])
        exposures = {}
        for sub in sub_sig_list:
            sub_sig = (pd.read_pickle(f'{self.dump_dir}/{sub}.pkl')
                       .query('@start <= index <= @end')[signal_col])
            exposures[sub] = (sub_sig * port_sig).mean()
        result = pd.Series(exposures, name='exposure')
        if verbose:
            for name, val in result.items():
                print(f'{name} exposure: {val:.2f}')
        return result

    def correlation_analysis(
        self,
        check_id: str,
        checklist: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        策略间相关性分析：对比指定策略与其他策略的日收益率相关性。

        Parameters
        ----------
        check_id : str
            基准策略 ID（文件名不含 .pnl.pkl 后缀）。
        checklist : list of str, optional
            对比策略 ID 列表。若为 None 则自动发现 pnl 目录下所有其他策略。
        start_date : str, optional
            只保留此日期之后的数据（YYYYMMDD）。
        verbose : bool
            是否打印结果。

        Returns
        -------
        pd.DataFrame
            完整相关性矩阵。
        """
        base = pd.read_pickle(f'{self.pnl_dir}/{check_id}.pnl.pkl')['pnl']
        base.name = check_id
        compare = [base]

        if checklist is None:
            checklist = sorted([
                f[:-8] for f in os.listdir(self.pnl_dir)
                if f.endswith('.pnl.pkl') and f[:-8] != check_id
            ])
        for pid in checklist:
            fpath = f'{self.pnl_dir}/{pid}.pnl.pkl'
            if not os.path.exists(fpath):
                continue
            s = pd.read_pickle(fpath)['pnl']
            s.name = pid
            compare.append(s)

        corr_df = pd.concat(compare, axis=1, join='inner')
        if start_date:
            corr_df = corr_df[pd.to_datetime(corr_df.index) >
                              datetime.strptime(start_date, '%Y%m%d')]
        corr_mat = corr_df.corr()
        if verbose:
            others = corr_df.drop(columns=[check_id]).corrwith(corr_df[check_id])
            print(others)
            print(corr_mat)
        return corr_mat

    # ========================================================================
    # Data utilities — 市场数据辅助
    # ========================================================================

    @staticmethod
    def _load_data_dict(data_list: List[str], start: str, end: str,
                        data_dir: str = './data') -> dict:
        """加载多个数据文件，返回 {name: DataFrame}。"""
        data_dict = {}
        for name in data_list:
            df = pd.read_pickle(f'{data_dir}/{name}.pkl').query('@start <= index <= @end')
            df = df.loc[:, ~df.columns.str.contains('BJ')]
            data_dict[name] = df
        return data_dict

    @classmethod
    def stock_crowd_index(cls, start: str, end: str,
                          data_dir: str = './data') -> pd.DataFrame:
        """
        计算个股资金拥挤度指标：成交额前 10% 的股票占总成交额的比例。

        Parameters
        ----------
        start, end : str
            统计区间（YYYYMMDD）。
        data_dir : str
            数据目录。

        Returns
        -------
        pd.DataFrame
            列：crowd。
        """
        data_dict = cls._load_data_dict(['close', 'amount'], start, end, data_dir)
        amt_rk = data_dict['amount'].rank(axis=1, pct=True)
        crowd = data_dict['amount'][amt_rk > 0.9].sum(1) / data_dict['amount'].sum(1)
        return pd.DataFrame({'crowd': crowd})


# ============================================================================
# Module-level aliases — 支持直接导入函数
# ============================================================================

structure_stats = StrategyAnalyzer.structure_stats
annual_stat = StrategyAnalyzer.annual_stat
annual_metric = StrategyAnalyzer.annual_metric
decile_analysis = StrategyAnalyzer.decile_analysis
rolling_stats = StrategyAnalyzer.rolling_stats
