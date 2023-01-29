""" 
This module was created to plot useful graphics
which will be mostly used in EDA.

"""


from drtools.utils import list_ops, to_title
from drtools.file_manager import create_directories_of_path
from drtools.data_science.data_handle import prepare_bins, binning_numerical_variable
from types import FunctionType
from typing import Dict, List, Tuple, Union
import pandas as pd
from typing import Any
from pandas import DataFrame, Series
from numpy import inf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
# import shap


COLORS = [
    {'PRIMITIVE': '#0D4C92', 'PRETTY': '#0D4C92'}, 
    {'PRIMITIVE': '#59C1BD', 'PRETTY': '#59C1BD'}, 
    {'PRIMITIVE': '#A0E4CB', 'PRETTY': '#A0E4CB'}, 
    {'PRIMITIVE': '#CFF5E7', 'PRETTY': '#CFF5E7'}, 
    
    {'PRIMITIVE': 'blue', 'PRETTY': 'cornflowerblue'}, 
    {'PRIMITIVE': 'red', 'PRETTY': 'indianred'}, 
    {'PRIMITIVE': 'green', 'PRETTY': 'springgreen'}, 
    {'PRIMITIVE': 'purple', 'PRETTY': 'magenta'}, 
]


def na_heatmap(
    dataframe: DataFrame,
    isna=True,
    figsize=(10, 5),
    vertical_sep: bool=False,
    rotate_x: int=45,
    nullity: bool=True,
    sort_columns: bool=True,
) -> None:    
    """Plot na distribution among columns of DataFrame. 

    Parameters
    ----------
    df : DataFrame
        DataFrame will be ploted
    isna : bool, optional
        If True, plot will show distribution of
        columns where na, else plot will show
        distribution of columns where not na, by default True
    figsize : Tuple[int, int], optional
        Size of graph, by default (10, 5)
    vertical_sep : bool, optional
        If True, will separate columns with a white line, 
        by default False.
    rotate_x : int, optional
        Value to rotate x labels, by default 45.    
    nullity : bool, optional
        If True, will display the NA percentage by column, 
        by default True.
    sort_columns : bool, optional
        If True, the columns of DataFrame will be sorted, 
        by default True.
        
    Returns
    ------
        None
            **None** is returned
    """
    
    operation = {
        True: 'isna',
        False: 'notna',
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df = dataframe.copy()
    
    if sort_columns:
        df = df.reindex(sorted(df.columns), axis=1)
    
    sns.heatmap(
        getattr(df, operation[isna])(),
        ax=ax
    )
        
    if vertical_sep:
        for i in range(df.shape[1] + 1):
            ax.axvline(i, color='white', lw=2)
        
    if rotate_x > 0:
        fig.autofmt_xdate(rotation=rotate_x)
        
    if nullity:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([i + 0.5 for i in range(df.shape[1])])        
        tick_labels = []
        for col in df.columns:
            
            percent = getattr(df[col], operation[isna])().sum() * 100 / df.shape[0]
            if 99 < percent < 100:
                percent = math.floor(percent)
            else:
                percent = math.ceil(percent)
                
            tick_labels.append(f'{percent}%')
        ax2.set_xticklabels(tick_labels)


def savefig(
    save_path: str
) -> None:
    """Save figure of subplot in selected path

    Parameters
    ----------
    save_path : str
        Path to save figure
        
    Returns
    ------
        None
            **None** is returned
    """
    plt.tight_layout()
    plt.savefig(save_path)


class Legend:
    """Class to handle the legends on plots
    
    """
    
    def __init__(
        self,
        active: bool=False,
        title: str='',
        labels: List[str]=[],
        loc='upper right',
        bbox_to_anchor: Tuple[float, float]=None,
    ) -> None:
        args = locals().copy()
        args = {k: v for k, v in args.items() if k != 'self'}
        for k, v in args.items():
            setattr(self, k, v)
            
    def prepare(self, subplot: plt) -> None:
        if not self.active:
            return None
        subplot.legend(
            title=self.title, 
            loc=self.loc, 
            labels=self.labels,
            bbox_to_anchor=self.bbox_to_anchor
        )
        return None


class TwinxValuesCalculation:
    
    def __init__(
        self,
        column: str=None,
        apply: FunctionType=None,
        agg: str=None,
        filter_function: FunctionType=None,
    ) -> None:
        self.column = column
        self.apply = apply
        self.agg = agg
        self.filter_function = filter_function


class Twinx:
    """Class to handle Twinx axe on the plots
    
    """
    
    def __init__(
        self,  
        active: bool=False,
        x: Union[List, Series]=[],
        y: Union[List, Series]=[],
        ylabel: str='Twinx Axe',
        ylim: Tuple[float, float]=(0, 1),
        color: str='tab:olive',
        label: str='',
        marker: str='o',
        grid: str=None,
        fillna: any=0,        
        legend: Legend=Legend(),
        twinx_values_calculation: TwinxValuesCalculation=TwinxValuesCalculation(),
        apply: List[FunctionType]=None,
    ) -> None:
        # if active \
        # and twinx_values_calculation.column is None:
        #     raise Exception(
        #         'If "active" is True, "twinx_values_calculation.column" need to be specified.'
        #     )  
        # if active \
        # and (
        #     twinx_values_calculation.apply is None
        #     and twinx_values_calculation.agg is None
        # ):
        #     text = 'If "active" is True, "twinx_values_calculation.apply" '
        #     text += 'or "twinx_values_calculation.agg" need to be specified.'
        #     raise Exception(text)  
             
        args = locals().copy()
        args = {k: v for k, v in args.items() if k != 'self'}
        for k, v in args.items():
            setattr(self, k, v)
            
    def prepare(self, subplot: plt) -> None:
        if not self.active:
            return None
        if type(self.x) == Series:
            self.x = self.x.values
        if type(self.y) == Series:
            self.y = self.y.values
        twinx = subplot.twinx()
        twinx.plot(
            self.x, 
            self.y, 
            color=self.color, 
            label=self.label, 
            marker=self.marker
        )
        if self.ylabel is not None:
            twinx.set_ylabel(self.ylabel)
        if self.ylim is not None:
            twinx.set_ylim(*self.ylim)
        twinx.grid(self.grid)        
        
        # twinx.set_yticks(
        #     np.linspace(
        #         twinx.get_yticks()[0], 
        #         twinx.get_yticks()[-1], 
        #         len(subplot.get_yticks())
        #     )
        # )
        
        self.legend.prepare(twinx)
        
        if self.apply is not None:
            for func in self.apply:
                func(twinx, subplot)
        
        return None


def bar(
    dataframe: DataFrame,
    analysis_variable: str,
    hue: str=None,
    figsize: Tuple[int, int]=(10, 5),
    fillna: float=0,
    na_as_class: bool=False,
    ignore_columns: List=[],
    sort_index_attr: dict={
        'ascending': True
    },
    sort_values_attr: dict={},
    rotate_x: int=0,
    twinx: Union[Twinx, List[Twinx]]=Twinx(),
    legend: Legend=Legend(
        active=True, loc='upper right'
    ),
    savefig_path: str='',
) -> None:
    """Plot custom stacked bar.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing both analysis variable and hue variable
    analysis_variable : str
        Variable to be ploted
    hue : str, optional
        Hue variable, by default None.
    figsize : Tuple[int, int], optional
        Size of graph, by default (10, 5)
    fillna : float, optional
        Value to fill NA values of DataFrame, by default 0
    na_as_class : bool, optional
        If True, NA values will be transformod in a class, 
        if False, NA values will not be considered when 
        ploting, by default False
    ignore_columns : List, optional
        Ignore these columns when plot data. These columns
        will not be considered to plot data, by default []
    sort_index_attr : dict, optional
        Attributes will be applied on sorting
        index, by default { 'ascending': True }
    sort_values_attr : dict, optional
        Attributes will be applied on sorting
        values, by default { }
    savefig_path : str, optional
        Path to save figure, by ''
    rotate_x : int, optional
        Value to rotate x labels, by default 0.    
    twinx : Union[Twinx, List[Twinx]], optional
        Attributes for the twinx axe, by default Twinx()
    legend : Legend, optional
        Attributes for the legend, by default Legend()
        
    Returns
    -------
    None
        **None** is returned
    """
    
    sns.set_style('darkgrid')
    df = dataframe.copy()
    
    if not hue:
        hue = '__temp_fill__'
        df[hue] = hue
        
    if na_as_class:
        df[analysis_variable] = df[analysis_variable].fillna('nan')
        
    margins_name = '__total__'
    
    df = pd.crosstab(
        df[analysis_variable], 
        df[hue],
        margins=True,
        margins_name=margins_name
    )
    df = df.fillna(fillna)
    
    work_columns = list_ops(df.columns, ignore_columns + [margins_name])

    fig, ax1 = plt.subplots(figsize=figsize)

    # Define Labels

    ax1.set_xlabel(analysis_variable)
    ax1.set_ylabel('Count')
    
    df.index = df.index.astype(str)
    df = df.sort_index(**sort_index_attr)
    if sort_values_attr:
        df = df.sort_values(**sort_values_attr)
    df = df.drop(margins_name, axis='index')
    df = df.drop(margins_name, axis='columns')
        
    df.index = df.index.map(str)

    bottom = None
    for idx, col in enumerate(work_columns):
        color = COLORS[idx % len(COLORS)]['PRETTY']
        if idx == 0:
            ax1.bar(
                df.index, 
                df[col], 
                color=color,
            )
            bottom = df[col]
        else:                    
            ax1.bar(
                df.index, 
                df[col], 
                bottom=bottom, 
                color=color
            )
            
            bottom = bottom + df[col]

    if hue != '__temp_fill__':
        legend.title = hue
        legend.labels = [to_title(str(col)) for col in work_columns]
        legend.prepare(ax1)
        
    # Twin Axes
    def plot_twinx(
        twinx_obj: Twinx, 
        subplot,
    ):
        if twinx_obj.active:
            
            index_len = len(df.index)
            twinx_obj.x = np.linspace(0, index_len - 1, index_len)
            
            twinx_values_calculation = twinx_obj.twinx_values_calculation
            values_from_column = twinx_values_calculation.column or hue
            apply_operation = twinx_values_calculation.apply
            agg_operation = twinx_values_calculation.agg
            filter_op = twinx_values_calculation.filter_function
            
            temp_df = dataframe.copy()
            
            if na_as_class:
                temp_df[analysis_variable] = temp_df[analysis_variable].fillna('nan')
                
            if filter_op is not None:
                twinx_obj.y = filter_op(temp_df.copy())
            else:
                twinx_obj.y = temp_df.copy()
                
            twinx_obj.y = twinx_obj.y.groupby([analysis_variable])
            twinx_obj.y = twinx_obj.y[values_from_column]
            if apply_operation is not None:
                twinx_obj.y = twinx_obj.y.apply(apply_operation)
            else:
                twinx_obj.y = twinx_obj.y.agg(agg_operation)
            twinx_obj.y = twinx_obj.y.reindex(index=list(df.index))
            twinx_obj.y = twinx_obj.y.values
            
            twinx_obj.prepare(subplot)
    
    if type(twinx) != list:
        twinx = [twinx]
        
    # if type(twinx) == list:
    for idx, curr_twinx in enumerate(twinx):
        if idx == 0:
            plot_twinx(twinx_obj=curr_twinx, subplot=ax1)
        else:
            color = COLORS[idx % len(COLORS)]['PRETTY']
            curr_twinx.color = color
            curr_twinx.ylabel = None
            curr_twinx.apply = [
                lambda twinx_plt, subplot: twinx_plt.set_yticklabels([])
            ]
            plot_twinx(twinx_obj=curr_twinx, subplot=ax1)
    # else:
    #     plot_twinx(
    #         twinx_obj=twinx, 
    #         subplot=ax1,                   
    #     )

    # Display
    plt.title(f'Bar plot of {analysis_variable}')
    
    if rotate_x > 0:
        fig.autofmt_xdate(rotation=rotate_x)
    
    if savefig_path:
        savefig(savefig_path)
        
    plt.show()


def hist(
    dataframe: DataFrame,
    analysis_variable: str,
    hue: str=None,
    analysis_range: Tuple[float, float]=(-inf, inf),
    figsize: Tuple[int, int]=(10, 5),
    bins: Union[int, List[float]]=20,
    kde: bool=False,
    extra_information: bool=True,
    alpha: float=0.4,
    mean: bool=False,
    twinx: Union[Twinx, List[Twinx]]=Twinx(fillna=0),
    legend: Legend=Legend(
        active=True, loc='upper right'
    ),
    savefig_path: str='',
) -> None:
    """Plot histogram with hue on binary variable

    Parameters
    ----------
    df : DataFrame
        DataFrame containing both analysis variable and hue variable
    analysis_variable : str
        Variable to be ploted
    hue : str
        Hue variable
    analysis_range : Tuple[float, float], optional
        Range of limits in which histogram
        will be plotted. Do not include upper range limit
        , by default (-inf, inf)
    figsize : Tuple[int, int], optional
        Size of ploted graph, by default (10, 5)
    bins : Union[int, List[float]], optional
        Number of bins or interval of values
        to binning ``analysis_variable``, by default 100
    kde : bool, optional
        Plot density, by default False
    extra_information : bool, optional
        Show extra information about ploted data, 
        by default True
    alpha : float, optional
        Alpha value of bins on hist plot. Must be between 0 and 1, by default 0.4
    mean : bool, optional
        Show mean on graph, by default False
    twinx : Union[Twinx, List[Twinx]], optional
        Attributes for the twinx axe, by default Twinx(fillna=0)
    legend : Legend, optional
        Attributes for the legend, by default Legend( active=True, loc='upper right' )
    savefig_path : str, optional
        Path to save figure, by ''   
        
    Returns
    ------
    None
        **None** is returned    
    """
    
    sns.set_style('darkgrid')
    df = dataframe.copy()
    original_total_points = len(df)
    
    temp_analysis_range = [None, None]
    temp_analysis_range[0] = max(bins[0], analysis_range[0]) \
        if hasattr(bins, '__iter__') \
        else analysis_range[0]
    temp_analysis_range[1] = min(bins[-1], analysis_range[1]) \
        if hasattr(bins, '__iter__') \
        else analysis_range[1]
    analysis_range = tuple(temp_analysis_range)
    lower_boundary = analysis_range[0]
    upper_boundary = analysis_range[1]
    
    try:
        v = df[analysis_variable].ge(lower_boundary) \
            & df[analysis_variable].lt(upper_boundary)
        df = df[v]
    except Exception as exc:
        print(f'Error when apply filters: {exc}')
        df = df.copy()        

    
    if not hue:
        hue = '__temp_fill__'
        df[hue] = hue
        
    v = df[hue].value_counts()
    series_data = {}
    for val in v.index:
        series_data[val] = df[df[hue] == val][analysis_variable]

    fig, ax1 = plt.subplots(figsize=figsize)
    
    # ax1.set_xlim([lower_boundary, upper_boundary])
    
    def mean_constructor(
        plt_instance,
        data_series: Series,
        mean_attr: Dict,
        annotate_position: int,
    ) -> None:
        plt_instance.vlines(
            data_series.mean(),
            **mean_attr
        )
        plt_instance.annotate(
            f'mean: {data_series.mean(): .2f}',
            xy=(data_series.mean() * 1.02, annotate_position)
        )
    
    # set bins
    sorted_values = df[analysis_variable] \
        .dropna() \
        .sort_values() \
        .values
    bins_interval, points, labels \
        = prepare_bins(
        bins,
        smallest_value=lower_boundary if lower_boundary != -inf else sorted_values[0], 
        bigger_value=upper_boundary if upper_boundary != inf else sorted_values[-1],
        include_upper=upper_boundary == inf
    )

    total_points = 0
    legend_labels = []
    for index, key in enumerate(series_data): 
        curr_series_data = series_data[key]
        curr_series_sorted = curr_series_data.sort_values()
        non_correspondences = pd.cut(
            curr_series_sorted,
            bins=bins_interval,
            right=False
        ).isna().sum()
        if non_correspondences != 0:
            total_points = total_points + len(
                curr_series_sorted.reset_index(drop=True)[:-non_correspondences]
            )
        else:
            total_points = total_points + len(
                curr_series_sorted
            )
        if hue != '__temp_fill__':
            legend_labels.append(f'{hue} is {str(key)}')
        sns.histplot(
            curr_series_data,
            color=COLORS[index % len(COLORS)]['PRETTY'],
            kde=kde,
            bins=bins_interval,
            alpha=alpha,
            ax=ax1
        )
        
    if hue != '__temp_fill__':
        legend.labels = legend_labels
        legend.prepare(ax1)

    if mean:
        max_value = -inf
        for index, key in enumerate(series_data): 
            curr_series_data = series_data[key]
            curr_max = pd.cut(
                curr_series_data.sort_values(),
                bins=bins_interval,
                right=False,
            ).value_counts().sort_values().values[-1]
            max_value = max(
                max_value, 
                curr_max
            )
        for index, key in enumerate(series_data):
            ser_data = series_data[key]
            real_max_value = max_value * 1.02
            mean_constructor(
                ax1,
                ser_data,
                {
                    'ymin': 0,
                    'ymax': real_max_value,
                    'color': COLORS[index % len(COLORS)]['PRIMITIVE']
                },
                real_max_value,
            )            
            
    # Twin Axes
    def plot_twinx(twinx_obj: Twinx, subplot):
        if twinx_obj.active:
            
            temp_df = binning_numerical_variable(
                df,
                analysis_variable,
                bins=bins_interval,
                binning_column_prefix='binning'
            )
            
            twinx_obj.x = points
            
            if len(twinx_obj.y) == 0:                
                twinx_values_calculation = twinx_obj.twinx_values_calculation
                values_from_column = twinx_values_calculation.column or hue
                apply_operation = twinx_values_calculation.apply
                agg_operation = twinx_values_calculation.agg
                filter_op = twinx_values_calculation.filter_function
                
                if filter_op is not None:
                    twinx_obj.y = filter_op(temp_df.copy())
                else:
                    twinx_obj.y = temp_df.copy()
                
                twinx_obj.y = twinx_obj.y.groupby([f'binning_{analysis_variable}'])
                twinx_obj.y = twinx_obj.y[values_from_column]
                if apply_operation is not None:
                    twinx_obj.y = twinx_obj.y.apply(apply_operation)
                else:
                    twinx_obj.y = twinx_obj.y.agg(agg_operation)
                twinx_obj.y = twinx_obj.y.values
            
            else:
                pass
            
            twinx_obj.prepare(subplot)
            
        
    if type(twinx) != list:
        twinx = [twinx]
        
    # if type(twinx) == list:
    for idx, curr_twinx in enumerate(twinx):
        if idx == 0:
            plot_twinx(twinx_obj=curr_twinx, subplot=ax1)
        else:
            color = COLORS[(idx-1) % len(COLORS)]['PRETTY']
            if curr_twinx.color == 'tab:olive':
                curr_twinx.color = color
            curr_twinx.ylabel = None
            curr_twinx.apply = [
                lambda twinx_plt, subplot: twinx_plt.set_yticklabels([])
            ]
            plot_twinx(twinx_obj=curr_twinx, subplot=ax1)
    # else:
    #     plot_twinx(
    #         twinx_obj=twinx, 
    #         subplot=ax1,                   
    #     )

    
    extra_information_text = ''
    if extra_information:
        trace_text = '-' * 100
        extra_information_text += f'{trace_text}'
        extra_information_text += '\n'
        extra_information_text += f'Information of {analysis_variable} variable'
        extra_information_text += '\n'
        extra_information_text += '\n'
        na_points = dataframe[analysis_variable].isna().sum()
        non_na_points = dataframe[analysis_variable].notna().sum()
        extra_information_text += f'<NA> points: '
        extra_information_text += f'{na_points:,} | '
        extra_information_text += f'{round(na_points * 100 / original_total_points, 3)}% '
        extra_information_text += f'#{original_total_points:,}'
        extra_information_text += '\n'
        extra_information_text += f'Non <NA> points: '
        extra_information_text += f'{non_na_points:,} | '
        extra_information_text += f'{round(non_na_points * 100 / original_total_points, 3)}% '
        extra_information_text += f'#{original_total_points:,}'
        extra_information_text += '\n'
        extra_information_text += f'Total Points on range '
        extra_information_text += f'[{analysis_range[0]:,}, {analysis_range[1]:,}): '
        extra_information_text += f'{total_points:,} | '
        extra_information_text += f'{round(total_points * 100 / non_na_points, 3)}% '
        extra_information_text += f'#{non_na_points:,}'
        if hue != '__temp_fill__':
            extra_information_text += '\n'
            for key in series_data:
                ser_data = series_data[key]
                text_value = str(key)
                curr_len = len(ser_data)
                percentage = round(curr_len * 100 / total_points, 2)
                extra_information_text += '\n'
                extra_information_text += f'- {hue} = {text_value} | {percentage}% #{curr_len:,}'
        extra_information_text += '\n'
        extra_information_text += f'{trace_text}'
        extra_information_text += '\n'
        extra_information_text += '\n'
    
    title_text = extra_information_text + f'Histogram of {analysis_variable}'
    
    ax1.set_title(title_text)
    ax1.set_xlabel(analysis_variable)
    
    if savefig_path:
        savefig(savefig_path)
        
    plt.show()


def roc_curve(
    y_true: List[float],
    y_score: List[float],
    pos_label: Union[int,str]=None,
    figsize: Tuple[int, int]=(5, 5),
    savefig_path: str='',
    **kwargs
) -> None:
    """Generate ROC Curve with AUC value.csv

    Parameters
    ----------
    y_true : List[float]
        True binary labels
    y_score : List[float]
        Target scores, can either be probability 
        estimates of the positive class, confidence 
        values, or non-thresholded measure of 
        decisions (as returned by “decision_function” 
        on some classifiers).
    pos_label : int, optional
        The label of the positive class. When 
        pos_label=None, if y_true is in {-1, 1} 
        or {0, 1}, pos_label is set to 1, 
        otherwise an error will be raised, 
        by default 1
    figsize : Tuple[int, int], optional
        Size of graph, by default (5, 5)
    savefig_path : str, optional
        Path to save figure, by ''
    kwargs : any, optional
        Extra arguments that will be passed to the 
        function metrics.roc_curve.
    """
    
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=y_true, 
        y_score=y_score, 
        pos_label=pos_label,
        **kwargs
    )
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_title('ROC Curve')
    ax1.plot(fpr, tpr, 'b', label='Model (AUC = %0.2f)' % roc_auc)
    ax1.plot([0, 1], [0, 1], 'r', label='Random')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('sensitivity')
    ax1.set_xlabel('1-specificity')
    
    if savefig_path:
        savefig(savefig_path)
    
    plt.show()
    
    
def lightgbm_feature_importance(
    model, 
    variables: List[str], 
    num=20, 
    figsize=(40, 20), 
    savefig_path: str=None
) -> None:
    """Plot importance from Light GBM model

    Parameters
    ----------
    model : Any
        The light gbm model instance
    variables : List[str]
        List of variables to verify importance.
    num : int, optional
        Num of variables to plot, by default 20
    figsize : Tuple[int, int], optional
        Size of graph, by default (40, 20)
    savefig_path : str, optional
        Path to save figure, by None
    """
    feature_imp = pd.DataFrame({
        'Value': model.feature_importance(),
        'Feature': variables
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # sns.set(font_scale=5)
    sns.barplot(
        x="Value", 
        y="Feature", 
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
        ax=ax
    )
    ax.set_title('LightGBM Features (avg over folds)')
    
    if savefig_path is not None:
        create_directories_of_path(savefig_path)
        savefig(savefig_path)
    plt.show()
    
    
def permutation_importance(
    model: Any,
    X: DataFrame,
    y: Series,
    n_repeats: int=3,
    random_state: int=42,
    n_jobs: int=2,
    figsize=(40, 20), 
    savefig_path: str=None
) -> None:
    """Plot permutation importance

    Parameters
    ----------
    model : Any
        Model instance
    X : DataFrame
        X data as DataFrame
    y : Series
        y data as Series
    n_repeats : int, optional
        Number of times to permute a feature, by default 3
    random_state : int, optional
        Pseudo-random number generator to control the 
        permutations of each feature. Pass an int to 
        get reproducible results across function calls, by default 42
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 2
    figsize : Tuple[int, int], optional
        Size of graph, by default (40, 20)
    savefig_path : str, optional
        Path to save figure, by None
    """
    
    # model_copy = deepcopy(model)
    
    # if 'predict' in dir(model_copy):
    #     setattr(model_copy, 'fit', model_copy.predict)
    
    result = sklearn_permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    sorted_importances_idx = result.importances_mean.argsort()
    importances = DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx]
    )
    
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances")
    ax.axvline(x=0, color='k', linestyle='--')
    ax.figure.tight_layout()
    
    if savefig_path is not None:
        create_directories_of_path(savefig_path)
        savefig(savefig_path)
        
    plt.show()
    

# def shap_importance(
#     model: Any,
#     X: DataFrame,
# ) -> None:
#     """Plot most important features using Shap Importance

#     Parameters
#     ----------
#     model : Any
#         Model instance
#     X : DataFrame
#         X data as DataFrame
#     """
#     # Explain model predictions using shap library:
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
    
#     # Plot summary_plot
#     shap.summary_plot(shap_values, X, plot_type='bar')
    
    
    
    
    
    