""" 
This module was created to handle time series stuff.

"""


from typing import List, Union
import drtools.utils as Utils
from pandas.core.frame import DataFrame
import pandas as pd
import math
import numpy as np
from drtools.logs import Log


OneDimension = int
TwoDimension = int
Axis0 = int
Axis1 = int


def shift(
    xs: np.ndarray, 
    periods: int, 
    fill_value: any=None,
    dimension: Union[OneDimension, TwoDimension]=1,
    axis: Union[Axis0, Axis1]=0
) -> np.ndarray:
    """Shift numpy array.

    Parameters
    ----------
    xs : np.ndarray
        The numpy array to shift.
    periods : int
        Number of periods to shift. Can be positive or negative.
    fill_value : any, optional
        The scalar value to use for newly introduced missing values, 
        by default None.

    Returns
    -------
    np.ndarray
        The shifted numpy array.
    """
    
    if dimension == 1:
        
        if periods >= 0:
            return np.concatenate((
                np.full(min(periods, len(xs)), fill_value), 
                xs[:-periods]
            ))
        
        else:
            return np.concatenate((
                xs[-periods:], 
                np.full(max(-periods, -len(xs)), fill_value)
            ))
        
    elif dimension == 2:
        
        if periods >= 0:
            
            if axis == 0:
                return np.concatenate(
                    (
                        np.full((min(periods, xs.shape[0]), xs.shape[1]), fill_value), 
                        xs[:-periods]
                    ),
                    axis=0
                )
            
            elif axis == 1:
                return np.concatenate(
                    (
                        np.full((xs.shape[0], min(periods, xs.shape[1])), fill_value), 
                        xs[:, :-periods]
                    ),
                    axis=1
                )
            
        else:
            
            if axis == 0:
                return np.concatenate(
                    (
                        xs[-periods:], 
                        np.full((min(abs(periods), xs.shape[0]), xs.shape[1]), fill_value)
                    ),
                    axis=0
                )
            
            elif axis == 1:
                return np.concatenate(
                    (
                        xs[:, -periods:], 
                        np.full((xs.shape[0], min(abs(periods), xs.shape[1])), fill_value)
                    ),
                    axis=1
                )
                
    raise Exception('Invalid "dimension" or "axis" option.')


def time_series_data_generator(
    dataframe: DataFrame,
    time_col: str,
    id_col: str,
    target_col: str,
    series_size: int,
    ignore_cols: List[str]=[],
    data_leak_cols: List[str]=[],
    immutable_cols: List[str]=[],
    light: bool=False,
    dtype: bool=True,
    chunksize: int=1,
    return_numpy: bool=False,
    LOGGER: Log=None
) -> DataFrame:
    """Generate data for time series prediction.py

    Parameters
    ----------
    dataframe : DataFrame
        DataFrame containing data that will be transformed 
        in time series.
    time_col : str
        Column that represent time of data.
    id_col : str
        Column to group data and generate time series
    target_col : str
        Predict column.
    series_size : int
        How many rows will be grouped on one row
    ignore_cols : List[str], optional
        List of columns to ignore when generating series data, 
        but the data about these columns are returned in the reponse without 
        any treatment or transformation, by default []
    data_leak_cols : List[str], optional
        Columns that are considered Data Leak, 
        by default []
    immutable_cols : List[str], optional
        Columns that not change accross time, by default []
    light : bool, optional
        If True will apply transformations on data 
        in the received DataFrame and do not will 
        create a copy of DataFrame, by default False.
    return_numpy : bool, optional
        If True will return the data as numpy array rather than 
        as DataFrame, this can safe memo usage, by default False.
    dtype : bool, optional
        If True, type cols after transformations.
        If False, will not type cols after transformations 
        , by default True.
    chunksize : int, optional
        Process by chunks, by default 1.
    LOGGER : Log, optional
        Logging object to debug execution, 
        by default None.
    
    Returns
    -------
    DataFrame
        The DataFrame after transform the data into Time Series.
    """
    
    LOGGER.debug(f'Preprocess data...')
    
    unique_ids_on_id_column = dataframe[id_col].unique()
    # num = math.ceil(len(unique_ids_on_id_column) / chunks)
    sub_ids = np.array_split(unique_ids_on_id_column, chunksize)
    sub_ids = [x for x in sub_ids if x.shape[0] > 0]
    sub_ids_len = len(sub_ids)
    
    
    if light:
        df = dataframe
    else:
        df = dataframe.copy()
    
    temp_columns = [id_col, time_col, target_col]
        
    real_features_columns = Utils.list_ops(
        df.columns,
        temp_columns + ignore_cols
    )
    
    df_dtypes = df.dtypes
    
    df.sort_values(by=[id_col, time_col], inplace=True)
    
    # df = df.sort_values(by=[id_col, time_col])
    
    real_ignore_cols = Utils.list_ops(ignore_cols, [id_col]) + [id_col]
    ignore_data_df = df.loc[:, real_ignore_cols]
    
    df = df.loc[:, [id_col, time_col] + real_features_columns + [target_col]]
    
    df_columns = df.columns
        
    mutable_columns = Utils.list_ops(
        real_features_columns + [target_col],
        immutable_cols
    )
    
    mutable_indexes = [df_columns.get_loc(col) for col in mutable_columns]
    
    columns_dtypes = {
        col: getattr(df_dtypes, col) for col in df_columns
    }
    ignore_data_columns_dtypes = {
        col: getattr(df_dtypes, col) for col in ignore_data_df.columns
    }
    
    inserted_columns = {}
    for i in range(series_size):
        inserted_columns = {
            **inserted_columns,
            **{
                f'{col}_n{i + 1}': getattr(df_dtypes, col) for col in mutable_columns
            }
        }
            
    columns_dtypes = {
        # **ignore_data_columns_dtypes,
        **columns_dtypes,
        **inserted_columns
    }
        
    final_columns = [col for col in columns_dtypes]
    
    final_values = None
    
    LOGGER.debug(f'Preprocess data... Done!')
    
    for chunck_num, ids in enumerate(sub_ids):
        
        LOGGER.debug(f'({chunck_num+1}/{sub_ids_len}) Process...')
        
        LOGGER.debug('Filtering data...')
        curr_df = df[df[id_col].isin(ids)]
        
        ignore_data = ignore_data_df[ignore_data_df[id_col].isin(ids)] \
            .drop(id_col, axis=1)
        ignore_data = ignore_data.values
        
        df_val = curr_df.values
            
        LOGGER.debug(f'Processing {curr_df.shape[0]:,} rows of data.')
        
        del curr_df
        LOGGER.debug('Filtering data... Done!')
        
    
        
        if LOGGER is not None:
            LOGGER.debug(f'Grouping by {id_col}...')
        
        df_grouped = np.split(
            df_val[:, [i for i in range(df_val.shape[1]) if i != id_col]], 
            np.unique(df_val[:, df_columns.get_loc(id_col)], return_index=True)[1][1:]
        )
        
        if LOGGER is not None:
            LOGGER.debug(f'Grouping by {id_col}... Done!')
            LOGGER.debug(f'Filling data...')
        
            
        for idx in range(len(df_grouped)):
            for i in range(series_size):
                temp_data = shift(
                    df_grouped[idx][:, mutable_indexes],
                    i + 1,
                    axis=0,
                    dimension=2
                )
                df_grouped[idx] = np.concatenate(
                    (df_grouped[idx], temp_data), 
                    axis=1
                )
                
        if LOGGER is not None:
            LOGGER.debug(f'Filling data... Done!')
            LOGGER.debug(f'Flating data...')
                
        df_val = np.array([group_data for group in df_grouped for group_data in group])
        del df_grouped
        
        if LOGGER is not None:
            LOGGER.debug(f'Flating data... Done!')
        
        
        if LOGGER is not None:
            LOGGER.debug(f'Generating final values array...')
        
        df_val = np.concatenate((ignore_data, df_val), axis=1)
        
        if final_values is None:
            final_values = df_val.copy()
        else:
            final_values = np.concatenate((final_values, df_val), axis=0)
        
        if LOGGER is not None:
            LOGGER.debug(f'Generating final values array... Done!')
            
        LOGGER.debug(f'({chunck_num+1}/{sub_ids_len}) Process... Done!')

    
    
    real_final_columns = ignore_cols + final_columns
    
    if len(data_leak_cols) > 0:
        if LOGGER is not None:
            LOGGER.debug(f'Removing data leak cols...')
            
        final_columns = Utils.list_ops(
            final_columns,
            data_leak_cols
        )
        
        dict_to_real_final_columns = {
            col: idx
            for idx, col in enumerate(real_final_columns)
        }
        
        real_final_columns = ignore_cols + final_columns
        
        final_cols_indexes = [
            dict_to_real_final_columns[col]
            for col in real_final_columns
        ]
        final_values = final_values[:, final_cols_indexes]
            
        if LOGGER is not None:
            LOGGER.debug(f'Removing data leak cols... Done!')
            
            
    if return_numpy:
        return final_columns, final_values
    else:
        LOGGER.debug(f'Generating final dataframe...')
        del df
        del ignore_data_df
        df = pd.DataFrame(
            final_values, 
            columns=real_final_columns
        )    
        del final_values
        LOGGER.debug(f'Generating final dataframe... Done!')
    
    if dtype:
        
        if LOGGER is not None:
            LOGGER.debug(f'Dtyping columns...')
        
        for idx, col in enumerate(df.columns):
            
            curr_idx = idx + 1
            LOGGER.debug(f'({curr_idx:,}/{df.columns.shape[0]:,}) Dtyping col {col}...')
            
            col_dtype = columns_dtypes.get(col, None)
            if col_dtype is None:
                LOGGER.debug(f'({curr_idx:,}/{df.columns.shape[0]:,}) Skipping Dtyping for col {col}...')
                continue
            try:
                df[col] = df[col].astype(col_dtype)
            except Exception as exc:
                
                text = f'Exception <{exc}> was generate when trying to apply ' 
                text += f'dtype {col_dtype} on column {col}' 
                
                if 'int' in col_dtype.name:
                    try:
                        df[col] = df[col].astype(pd.Int64Dtype())
                    except Exception as exc:
                        text = '1) ' + text    
                        text += f'\n2) Exception <{exc}> was generate when trying to apply ' 
                        text += f'dtype {pd.Int64Dtype()} on column {col}'              
                        LOGGER.error(text)
                else:
                    LOGGER.error(text)
                    
            LOGGER.debug(f'({curr_idx:,}/{df.columns.shape[0]:,}) Dtyping col {col}... Done!')
                    
        if LOGGER is not None:
            LOGGER.debug(f'Dtyping columns... Done!')            
        
    return df


def time_series_data_split(
    dataframe: DataFrame,
    time_col: str,
    target_col: str,
    sizes: List[float]=[0.5, 0.2, 0.3],
    light: bool=False,
    drop_time_col: bool=False,
) -> List[DataFrame]:
    """Split time series data in train, val, holdout, etc.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe representing the time series data.
    time_col : str
        Column that represent time of data.
    target_col : str
        Predict column.
    sizes : List[float], optional
        Percentage of splited datasets 
        , the sum of values must be equal 
        1, by default [0.5, 0.2, 0.3].
    light : bool, optional
        If True will apply transformations on data 
        in the received DataFrame and do not will 
        create a copy of DataFrame, by default False.
    drop_time_col : bool, optional
        If True, drop time column before return, 
        by default False.

    Returns
    -------
    List[DataFrame]
        List of X and y dataframes.
    """
    
    if light:
        df = dataframe
    else:
        df = dataframe.copy()
    
    index_split = []
    
    df.sort_values(by=time_col, inplace=True)
    # df = df.sort_values(by=time_col)
    
    # del df
    
    df.reset_index(drop=True, inplace=True)
    # df = df.reset_index(drop=True)
    
    curr_size = 0
    curr_index = 0
    for size in sizes:
        temp_index = math.ceil(df.shape[0] * (curr_size + size))
        index_split.append(
            (curr_index, temp_index)
        )
        curr_index = temp_index + 1
        curr_size = curr_size + size        
    
    if drop_time_col:
        df = df.drop(time_col, axis=1)        
    
    filters = []
    for idx_start, idx_end in index_split:
        filters.append(
            df.loc[idx_start:idx_end].index
        )
        
    # Split data
    
    resp = []
    
    for filter_ in filters:
        resp.append(
            df[df.index.isin(filter_)].drop(columns=[target_col])
        )
        resp.append(
            df[df.index.isin(filter_)][target_col]
        )
        # df = df[~df.index.isin(filter_)]
        
    del df
    
    return resp