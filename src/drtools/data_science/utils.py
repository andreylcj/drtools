""" 
This module was created to handle DataFrames and returns 
pandas DataFrame or pandas Series.

"""


from typing import List, Union, Tuple
from pandas import DataFrame, cut
import numpy as np
from copy import deepcopy


def get_labels_from_bin_interval(
    bins: List[float],
) -> List[str]:
    """Get bins labels from bins interval of values

    Parameters
    ----------
    bins : List[float]
        List containing bins boundary

    Returns
    -------
    List[str]
        The list with labels for each interval
    """
    return [f'{x}_{y}' for x, y in zip(bins[:-1], bins[1:])]


def prepare_bins(
    bins: Union[int, List[float]],
    smallest_value: float=None,
    bigger_value: float=None,
    include_upper: bool=False
) -> Tuple[List[float], List[float], List[str]]:
    """Generate values of bins, middle points and labels for the bins
    
    This function will generate points which represents the x axis
    for the bins, the middle points of the intervals of the bins and
    the labels for each bin.

    Parameters
    ----------
    bins : Union[int, List[float]]
        The desired number of bins that will be generated
        or the bins interval
    smallest_value : float, optional
        Smallest value from the interval, by default None.
    bigger_value : float, optional
        Bigger value from the interval, by default None.
    include_upper : bool, optional
        Whether the last interval should 
        be right-inclusive or not, by default False

    Returns
    -------
    Tuple[List[float], List[float], List[str]]
        Returns the values which represents the intervals 
        from the x axis, the middle points of these intervals and
        the labels for the bins interval.
    """
    values = deepcopy(bins) \
        if hasattr(bins, '__iter__') \
        else np.linspace(smallest_value, bigger_value, bins + 1)
    if include_upper:
        upper_boundary = round(values[-1] * (1 + 0.001), 4) \
            if values[-1] != 0 \
            else 1e-4
        values[-1] = upper_boundary
    middle_points = [(x + y) / 2 for x, y in zip(values[:-1], values[1:])]
    labels = get_labels_from_bin_interval(values)
    labels = [
        f'[{label})'.replace('_', ', ')
        for label in labels
    ]
    return values, middle_points, labels
    

def binning_numerical_variable(
    dataframe: DataFrame,
    col_name: str,
    bins: Union[int, List[float]],
    binning_column_prefix: str='binning',
    include_upper: bool=False,
    labels: List[str]=None
) -> DataFrame:
    """Binning numerical variable.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame from where numerical variable will be found.
    col_name : str
        The numerical column name.
    bins : Union[int, List[float]]
        Desired number of bins or interval
        of bins
    binning_column_prefix : str, optional
        Prefix of new columns that will be generated
        with the binning of numerical variable, by default 'binning'
    include_upper : bool
        Whether the last interval should 
        be right-inclusive or not, by default False
    labels : List[str]
        Labels for the bins interval, by default **None**

    Returns
    -------
    DataFrame
        The DataFrame with new column containing the binned values
        for the numerical variable
    """
    
    df = dataframe.copy()
        
    sorted_values = df[col_name] \
        .dropna() \
        .sort_values() \
        .values
    
    smallest_value = sorted_values[0]
    bigger_value = sorted_values[-1]
    values, middle_points, _labels = prepare_bins(
        bins,
        smallest_value=smallest_value,
        bigger_value=bigger_value,
        include_upper=include_upper
    )
    df[f'{binning_column_prefix}_{col_name}'] = cut(
        df[col_name], 
        bins=values,
        labels=labels or _labels,
        right=False
    )
    
    return df


def reduce_mem_usage(
    dataframe: DataFrame, 
    verbose=True
):
    """Reduce memory usage from DataFrame
    
    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame.
    verbose : bool, optional
        If True, will print reduction information, 
        by default True.

    Returns
    -------
    DataFrame
        The resulting DataFrame after reduce memory usage.
    """    
    
    df = dataframe.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print('Memory usage BEFORE optimization is: {:.2f} MB'.format(start_mem))
        print('Memory usage AFTER optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df