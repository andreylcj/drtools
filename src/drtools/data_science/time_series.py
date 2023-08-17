""" 
This module was created to handle time series stuff.

"""


from typing import List
from drtools.utils import split_into_chunks
from pandas import DataFrame, Series
import pandas as pd
from drtools.logging import Logger, FormatterOptions
from datetime import datetime


def add_previous_values_by_group(
    dataframe: DataFrame, 
    group_column: str,
    time_column: str, 
    window_size: int,
    columns: List[str],
    light: bool=False,
    window_absolute_value: bool=False,
    sort_values: bool=True,
    chunksize: int=1,
    LOGGER: Logger=Logger(
        name="AddPreviousValuesByGroup",
        formatter_options=FormatterOptions(
            include_datetime=True,
            include_thread_name=True,
            include_logger_name=True,
            include_level_name=True
        ),
        default_start=False
    )
):
    if light:
      df: DataFrame = dataframe
    else:
      df: DataFrame = dataframe.copy()

    if sort_values:
      df.sort_values(by=[group_column, time_column], inplace=True)

    original_index = df.index
    unique_group_col_values: List = df[group_column].unique()

    group_col_chunks: List[List] \
      = split_into_chunks(unique_group_col_values, chunksize)

    total_chunks: int = len(group_col_chunks)

    final_df: DataFrame = None

    start_range: int = 1 if not window_absolute_value else window_size

    for idx, group_col_chunk in enumerate(group_col_chunks):
        curr_idx: int = idx + 1
        started_at: datetime = datetime.now()

        LOGGER.debug(f'({curr_idx:,}/{total_chunks:,}) Computing chunk...')

        work_df: DataFrame = df[df[group_column].isin(group_col_chunk)].copy()
        df: DataFrame = df[~df[group_column].isin(group_col_chunk)]
        
        LOGGER.debug(f'Chunk Shape: ({work_df.shape[0]:,}, {work_df.shape[1]:,})')
        LOGGER.debug(f'Remaining Data Shape: ({df.shape[0]:,}, {df.shape[1]:,})')

        new_column_names: List[str] = []
        new_columns: List[Series] = []

        for i in range(start_range, window_size + 1):
            for col in columns:
                col_name: str = f"{col}_n-{i}"
                new_column_names.append(col_name)
                new_columns.append(work_df.groupby(group_column)[col].shift(i))

        new_df: DataFrame = pd.concat(new_columns, axis=1)
        new_df.columns: List[str] = new_column_names
        work_df: DataFrame = pd.concat([work_df, new_df], axis=1)

        if final_df is None:
            final_df: DataFrame = work_df.copy()
        
        else:
            final_df: DataFrame = pd.concat([
                    final_df,
                    work_df
                ],
                axis=0,
                ignore_index=True
            )
            
        LOGGER.debug(f'Final Data Shape: ({final_df.shape[0]:,}, {final_df.shape[1]:,})')

        duration: float = round((datetime.now() - started_at).total_seconds(), 2)
        LOGGER.debug(f'({curr_idx:,}/{total_chunks:,}) Chunk computation ends in {duration}s.')

    del df

    final_df.set_index(original_index, inplace=True)

    return final_df