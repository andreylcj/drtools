""" 
This module was created to handle DataFrames and returns 
pandas DataFrame or pandas Series.

"""


import drtools.utils as Utils
from typing import List, Union, Tuple
from pandas import DataFrame, Series, cut
from scipy import stats
import numpy as np
import math
from copy import deepcopy


ByPercentage = 'by-percentage'
ByCount = 'by-count'
Lower = 'lower'
Upper = 'upper'


def combine_variables(
    dataframe: DataFrame,
    varibles: List[str],
    sep: str='|'
) -> DataFrame:
    """Combine categorical variables of DataFrame in order to analyse 
    impact of combined variables on response.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame
    varibles : List[str]
        List of categorical variables that will be combined
    sep : str, optional
        Separation character when combine variables, 
        by default '|'

    Returns
    -------
    DataFrame
        The DataFrame with new colum containing all combinations 
        of desired variables.
    """
    
    df = dataframe.copy()
    title = sep.join(varibles)
    def get_all_values(x):
        text = ''
        variables_len = len(varibles)
        for idx, var in enumerate(varibles):
            text += str(x[var])
            if idx != variables_len - 1:
                text += str(sep)
        return text
    df[title] = df.apply(get_all_values, axis=1)
    return df


def remove_boundary_values(
    dataframe: DataFrame,
    by: Union[ByPercentage, ByCount]=ByPercentage,
    value: float=0.05,
    boundary: Union[Lower, Upper]=Upper
) -> DataFrame:
    """Remove values of top part of the DataFrame

    Parameters
    ----------
    df : DataFrame
        The DataFrame
    by : Union[ByPercentage, ByCount], optional
        Rule that will be applied to remove top
        values, by default ByPercentage
    value : float, optional
        The value of the rule, by default 0.05
    boundary : Union[Lower, Upper], optional
        Extremity from where values 
        will be removed, by default 0.05

    Returns
    -------
    DataFrame
        The DataFrame after removing selected values of the 
        top part of DataFrame

    Raises
    ------
    ValueError
        If **by** option is invalid.
    """
    
    methods = {
        Lower: {
            ByPercentage: math.floor(value * len(dataframe)),
            ByCount: value,
            'df_filter': lambda x: dataframe.iloc[x:]
        },
        Upper: {
            ByPercentage: math.ceil((1-value) * len(dataframe)),
            ByCount: value,
            'df_filter': lambda x: dataframe.iloc[:x]
        }
    }    
    limit = methods[boundary][by]
    df_copy = methods[boundary]['df_filter'](limit)
    return df_copy


def remove_outliers(
    df: DataFrame,
    col_name: str,
    zscore_limit: int=3,
    fillna: any=None
) -> DataFrame:
    """Remove outliers using zscore rule
    
    Only will be keeped values with zscore lower than **zscore_limit**

    Parameters
    ----------
    df : DataFrame
        The DataFrame
    col_name : str
        The column name where outliers will be removed
    zscore_limit : int, optional
        Z-Score limit, by default 3
    fillna : any, optional
        Value to fill na before apply the zscore rule, by default None

    Returns
    -------
    DataFrame
        The DataFrame after removing the outliers
    """
    df_copy = df.copy()
    if fillna is not None:
        df_copy[col_name] = df_copy[col_name].fillna(fillna)
    df_copy = df[
        np.abs(stats.zscore(df_copy[col_name])) < zscore_limit
    ]
    return df_copy


ColumnInfo = dict({
    str: float
})


def keep_categories(
    df: DataFrame,
    col_name: str,
    keep_categories: List[str]=[]
) -> DataFrame:
    """Keep only selected categories from column of DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame
    col_name : str
        The column name where categories will be keeped
    keep_categories : List[str], optional
        List of categories that will be keeped, by default []

    Returns
    -------
    DataFrame
        The DataFrame after remove unselected categories and keeped
        selected categories
    """
    df_copy = df[col_name].isin(
        keep_categories
    )
    return df[df_copy]


def join_categories(
    dataframe: DataFrame,
    col_name: str,
    categories: List[str]=[], 
    to_category: str='others'
) -> DataFrame:
    """Join categories and fill these values with desired value.

    Parameters
    ----------
    df : DataFrame
        The DataFrame
    col_name : str
        The column name where categories will be joined.
    categories : List[str], optional
        List of categories will be joined, by default []
    to_category : str, optional
        New value for selected categories, by default 'others'

    Returns
    -------
    DataFrame
        The DataFrame after join the categories
    """
    df = dataframe.copy()
    if len(categories) == 0:
        return df
    condition = df[col_name].isin(categories)
    df.loc[condition, col_name] = to_category
    return df


def drop_categories(
    dataframe: DataFrame,
    col_name: str,
    categories: List[str],
) -> DataFrame:
    """Drop categories of column

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame
    col_name : str
        The column name where categories will be dropped
    categories : List[str]
        List of categories that will be dropped

    Returns
    -------
    DataFrame
        The DataFrame after drop the categories
    """
    condition = ~dataframe[col_name].isin(
        categories
    )
    return dataframe[condition]


def get_categories_ge(
    df: DataFrame,
    col_name: str,
    by: Union[ByPercentage, ByCount]=ByPercentage,
    value: float=0.01,
    ignore_categories: List[str]=[]
) -> List[str]:
    """Get name of categories that contains percentage or counts 
    of total samples greater or equal than value.

    Parameters
    ----------
    df : DataFrame
        DataFrame where categoreis will be found
    col_name : str
        Column name where categories will be found
    by : Union[ByPercentage, ByCount], optional
        Rule that will be applied to get categories based
        on percentage of total or count, by default ByPercentage
    value : float, optional
        The value of the rule, by default 0.01
    ignore_categories : List[str], optional
        Ignore this categories when apply percentage 
        filter, by default []

    Returns
    -------
    List[str]
        The list of categories with at least desired percentage of total
    """
    resp = []
    v = df[col_name].value_counts()
    col_len = df[col_name].notna().sum()
    
    for category, category_count_value in zip(v.index, v.values):
        if category in ignore_categories:
            continue
        
        if by == ByPercentage:
            compare_value = category_count_value / col_len
        elif by == ByCount:
            compare_value = category_count_value
        else:
            raise ValueError('Invalid "by" option.')       
        
        if compare_value >= value:
            resp.append(category)
    return resp


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


def join_categories_lt(
    dataframe: DataFrame,
    col_name: str,
    by: Union[ByPercentage, ByCount]=ByPercentage,
    value: float=0.01,
    to_category_name: str='others',
) -> DataFrame:
    """Fill value with 'to_category_name' of categories 
    which has count lower than 'value' in count or percentage

    Parameters
    ----------
    dataframe : DataFrame
        DataFrame
    col_name : str
        Name of categorical column
    by : Union[ByPercentage, ByCount], optional
        Rule that will be applied to join categories based
        on percentage of total or count, by default ByPercentage
    value : float, optional
        The value of the rule, by default 0.01
    to_category_name : str, optional
        Value to fill categories, by default 'others'

    Returns
    -------
    DataFrame
        DataFrame after applied changes. 
    """
    df = dataframe.copy()
    keep_categories = get_categories_ge(
        df, 
        col_name, 
        by=by,
        value=value,
    )
    all_categories = list(df[col_name].value_counts().index)
    join_categories_list = Utils.list_difference(
        all_categories, 
        keep_categories
    )
    df = join_categories(
        df, 
        col_name, 
        categories=join_categories_list,
        to_category=to_category_name
    )
    return df


def keep_categories_ge(
    df: DataFrame,
    col_name: str,
    by: Union[ByPercentage, ByCount]=ByPercentage,
    value: float=0.01,
    force_remove: List[str]=[]
) -> DataFrame:
    """Keep categories of 'col_name' which contains 
    percentage or counts greater or equal than 'value'.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame
    col_name : str
        Column name to keep categories
    by : Union[ByPercentage, ByCount], optional
        Rule that will be applied to keep categories based
        on percentage of total or count, by default ByPercentage
    value : float, optional
        Keep categories which has at least
        this percentage or count of total 
        samples, by default 0.01
    force_remove : List[str], optional
        Remove columns from this list even if this columns
        doesn't match the percentage condition, by default []

    Returns
    -------
    DataFrame
        DataFrame with only desired categories
    """
    
    keep_categories = get_categories_ge(
        df, 
        col_name, 
        by=by,
        value=value,
    )
    keep_categories = Utils.list_difference(
        keep_categories, 
        force_remove
    )
    df_copy = keep_categories(
        df, 
        col_name, 
        keep_categories=keep_categories
    )
    return df_copy


def na_percentage_of_column(
    df: DataFrame,
    col_name: str,
) -> float:
    """Number representing NA percentage from column of DataFrame 

    Parameters
    ----------
    df : DataFrame
        The DataFrame
    col_name : str
        Columns name to verify nullity

    Returns
    -------
    float
        The percentage of na for the columns passed as 'col_name'
        argument
    """
    return df[col_name].isna().sum() / len(df[col_name])


def percentage_of_most_common_value_by_column(
    df: DataFrame,
    ignore_columns: List[str]=[],
    notna: bool=False
) -> ColumnInfo:
    """Percentage of the most common value for each column.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame
    ignore_columns : List[str], optional
        Ignore columns when get percentage, by default []
    notna : bool, optional
        If True, percentage will be based on not na values,
        if False, percentage will be calculated considering na
        values, by default False

    Returns
    -------
    ColumnInfo
        Dict containing information about percentage of most common
        value from each column.
    """
    
    resp = {}
    for col in df.columns:
        if col in ignore_columns:
            continue
        
        data_series = df[col].dropna() \
            if notna \
            else df[col].fillna('nan')
        
        v = data_series.value_counts()
        num_of_occ = v.sort_values().values
        num_of_occ = num_of_occ[-1]
        total_len = len(data_series)
        value_percentage = num_of_occ / total_len
        
        resp[col] = round(value_percentage * 100, 2)
    return resp


def na_percentage_info(
    df: DataFrame,
    ignore_columns: List[str]=[]
) -> ColumnInfo:
    """Get percentage of null of all columns.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame.
    ignore_columns : List, optional
        Ignore these columns when check DataFrame, by default []

    Returns
    -------
    ColumnInfo
        Information of null percentage about each column
    """
    
    resp = {}
    for col in df.columns:
        if col not in ignore_columns:
            resp[col] = round(
                na_percentage_of_column(df, col) * 100, 
                2
            )
    return resp


def get_column_names(
    df: DataFrame,
    contains: List[str]=[]
) -> List[str]:
    """Get name of columns that contains some of substrings in 'contains'

    Parameters
    ----------
    df : DataFrame
        DataFrame from which columns will be generated
    contains : List[str], optional
        List of substring to filter DataFrame columns, by default []

    Returns
    -------
    List[str]
        The list of column names which has some of the substrings
        in 'contains'
    """
    
    names = []
    if len(contains) == 0:
        contains = ['']
    for col in df.columns:
        for col_name in contains:
            if col_name in col:
                names.append(col)
                break
    return names


def drop_df_columns_gt_constancy(
    dataframe: DataFrame,
    constacy_level: float=0.75,
    verbose: bool=True,
    print_removed_columns: bool=False,
) -> DataFrame:
    """Drop columns based on variable constancy.
    
    Drop columns that has some class or value with count percentage 
    bigger than constancy_level.

    Parameters
    ----------
    dataframe : DataFrame
        DataFrame that columns will be dropped
    constacy_level : float, optional
        Columns that has some value with percentage of counts 
        above constancy level will be dropped, must be
        between 0 and 1, by default 0.8
    verbose : bool, optional
        If True print extra information, by default True
    print_removed_columns : bool, optional
        If True print removed columns, by default False

    Returns
    -------
    DataFrame
        DataFrame after drop columns
    """
    df = dataframe.copy()
    drop_columns = []
    for col in df.columns:
        v = df[col].value_counts()
        num_of_occ = v.sort_values().values[-1]
        total_len = len(df[col])
        variable_constancy_level = num_of_occ / total_len
        if variable_constancy_level >= constacy_level:
            drop_columns.append(col)
    df.drop(columns=drop_columns, axis=1, inplace=True)
    if verbose:
        len_initial = len(dataframe.columns)
        len_final = len(df.columns)
        print(f'From {len_initial} to {len_final}. Remove: {len_initial - len_final}')
    if print_removed_columns:
        removed = [col for col in df.columns if col not in df.columns]
        print(removed)
    return df


def merge_columns(
    dataframe: DataFrame,
    cols: List[str],
) -> Series:
    """Create pandas Series containing the result of 
    combination of some list of columns.
    
    The resulting Series will contain the first filled value 
    from a column passed on variable `cols`.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame.
    cols : List[str]
        List of column names.

    Returns
    -------
    Series
        The resulting Series.
    """
    
    df = dataframe[cols].copy()
    series = dataframe[cols[0]]
    
    for col in cols[1:]:
        series = series.fillna(df[col])
    
    return series


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