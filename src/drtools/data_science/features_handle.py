""" 
This module was created to handle Features construction and 
other stuff related to features from Machine Learn Model.

"""


from drtools.utils import (
    start_end_log, list_ops
)
from pandas import DataFrame
import pandas as pd
from typing import List, Union, Dict


ColumnName = str
EncodeValue = List[Union[str, int]]


@start_end_log('single_ohe')
def single_ohe(
    dataframe: DataFrame,
    column: str,
    encode_values: List[EncodeValue],
    prefix: str=None,
    prefix_sep: str="_",
    drop_self_col: bool=True
) -> DataFrame:
    """One hot encode one column, drop original column after 
    generate encoded and drop dummy cols that is not desired on 
    final data.
    
    Parameters
    ----------
    dataframe : DataFrame
        DataFrame containing data to encode.
    column : str
        Name of column to one hot encode.
    encode_values: List[Union[str, int]]
        List of values to encode.
    prefix: str, optional
        Prefix of encoded column. If None, 
        the prefix will be the column name, by default None.
    prefix_sep: str, optional
        Separation string of Prefix and Encoded Value, 
        by default "_".
    drop_self_col: bool, optional
        If True, the encoded column will be deleted. 
        If False, the encoded column will not be deleted, 
        by default True.
        
    Returns
    -------
    DataFrame
        The DataFrame containing encoded columns.
    """
    if prefix is None:
        prefix = column    
    finals_ohe_cols = [
        f'{prefix}{prefix_sep}{x}'
        for x in encode_values
    ]
    df = dataframe.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix, prefix_sep= prefix_sep)
    drop_cols = list_ops(dummies.columns, finals_ohe_cols)
    df = pd.concat([df, dummies], axis=1)
    if drop_self_col:
        drop_cols = drop_cols + [col]
    df = df.drop(drop_cols, axis=1)            
    # insert feature that not has on received dataframe
    for col in finals_ohe_cols:
        if col not in df.columns:
            df[col] = 0
    return df

@start_end_log('one_hot_encoding')
def one_hot_encoding(
    dataframe: DataFrame,
    encode: Dict[ColumnName, List[EncodeValue]],
    drop_self_col: bool=True
) -> DataFrame:
    """One hot encode variables, drop original column that 
    generate encoded and drop dummy cols that is not present 
    on the input features.
    
    Parameters
    ----------
    dataframe : DataFrame
        DataFrame containing data to encode.
    encode : Dict[ColumnName, List[EncodeValue]]
        The dict containing column names and values to encode.
    drop_self_col: bool, optional
        If True, the encoded column will be deleted. 
        If False, the encoded column will not be deleted, 
        by default True.
        
    Returns
    -------
    DataFrame
        The DataFrame containing encoded columns.
    """
    df = dataframe.copy()
    for column_name, encode_values in encode.items():
        df = single_ohe(
            dataframe=df,
            column=column_name,
            encode_values=encode_values,
            drop_self_col=drop_self_col
        )
    return df