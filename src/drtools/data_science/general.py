""" 
This module was created to archive general functions, methods, 
classes and other stuff.

"""


from typing import Dict, List, Union, Optional, TypedDict, Callable
import numpy as np
from pandas import DataFrame
import pandas as pd
from drtools.utils import list_ops
from drtools.logs import Log
import ast
# from drtools.data_science.features_handle import FeatureJSON


### Comparison Operators
EqualsTo = "$eq"
GreaterThan = "$gt"
GreaterThanOrEqual = "$gte"
In = "$in"
LessThan = "$lt"
LessThanOrEqual = "$lte"
NotEqual = "$ne"
NotIn = "$nin"

class ComparisonQueryOperators(TypedDict):
    EqualsTo: Optional[Union[float, str]]
    GreaterThan: Optional[float]
    GreaterThanOrEqual: Optional[float]
    In: Optional[List[Union[float, str]]]
    LessThan: Optional[float]
    LessThanOrEqual: Optional[float]
    NotEqual: Optional[Union[float, str]]
    NotIn: Optional[List[Union[float, str]]]


FieldName = str
FieldQuery = Dict[FieldName, ComparisonQueryOperators]


### Logical Operators
And = "$and"
Or = "$or"

class LogicalQueryOperators(TypedDict):
    And: Optional[Union[List, List[FieldQuery]]]
    Or: Optional[Union[List, List[FieldQuery]]]


Query = Union[LogicalQueryOperators, ComparisonQueryOperators]


NumpyDataType = 'numpy'
DataFrameDataType = 'dataframe'
Data = Union[DataFrame, np.array]


class FindOnData:
    """Query for data on Numpy matrix or Pandas DataFrame.
    """
    
    def __init__(
        self,
        data: Data,
        query: Query,
        data_type: Union[DataFrameDataType, NumpyDataType]=DataFrameDataType,
        data_columns: List[str]=None,
        LOGGER=None
    ) -> None:        
        assert data_type in [DataFrameDataType, NumpyDataType], \
            f'Invalid data_type: {data_type}'
        if data_type == NumpyDataType:
            assert data_columns is not None and len(data_columns) > 0, \
                f'When data_type equals to {data_type}, "data_columns" must be provided.'
        self.Query = query
        self.Data = data
        self.DataType = data_type
        self.DataColumns = data_columns
        if self.DataColumns is not None:
            self.DataColNameToIdx = {
                col_name: idx
                for idx, col_name in enumerate(self.DataColumns)
            }
        if LOGGER is None:
            self.LOGGER = Log()
        else:
            self.LOGGER = LOGGER
    
    def _is_valid_logical_syntax(self, query) -> bool:
        try:
            if And in query:
                if Or in query:
                    return False
                if isinstance(query[And], list):
                    return True
                else:
                    return False
            elif Or in query:
                if isinstance(query[Or], list):
                    return True
                else:
                    return False
            else:
                keys_list = list(query.keys())
                if len(keys_list) == 0:
                    return True
                else:
                    return False
        except:
            return False
    
    def _is_valid_comparison_syntax(self, query) -> bool:
        if not isinstance(query, dict):
            return False
        
        try:
            for k, v in query.items():
                keys_list = list(v.keys())
                not_expected_keys = list_ops(
                    keys_list,
                    [
                        EqualsTo, GreaterThan, GreaterThanOrEqual, 
                        In, LessThan, LessThanOrEqual, NotEqual, NotIn
                    ]
                )
                if len(not_expected_keys) > 0:
                    return False
            return True
        except:
            return False
    
    
    ##########################################
    ### Comparison Operators
    ##########################################
    def _is_eq_op(self, op_name):
        return op_name == "$eq"
    
    def _is_gt_op(self, op_name):
        return op_name == "$gt"
    
    def _is_gte_op(self, op_name):
        return op_name == "$gte"
    
    def _is_in_op(self, op_name):
        return op_name == "$in"
    
    def _is_lt_op(self, op_name):
        return op_name == "$lt"
    
    def _is_lte_op(self, op_name):
        return op_name == "$lte"
    
    def _is_ne_op(self, op_name):
        return op_name == "$ne"
    
    def _is_nin_op(self, op_name):
        return op_name == "$nin"
    
    def _is_isna_op(self, op_name):
        return op_name == "$isna"
    
    def _perform_eq_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return self.Data[column] == value
        elif self.DataType == NumpyDataType:
            return self.Data[:, column] == value
    
    def _perform_gt_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return self.Data[column] > value
        elif self.DataType == NumpyDataType:
            return self.Data[:, column] > value
    
    def _perform_gte_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return self.Data[column] >= value
        elif self.DataType == NumpyDataType:
            return self.Data[:, column] >= value
    
    def _perform_in_op(self, column, value: List):
        if self.DataType == DataFrameDataType:
            return self.Data[column].isin(value)
        elif self.DataType == NumpyDataType:
            raise Exception("Operation $in must for NumpyDataType.")
    
    def _perform_lt_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return self.Data[column] < value
        elif self.DataType == NumpyDataType:
            return self.Data[:, column] < value
    
    def _perform_lte_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return self.Data[column] <= value
        elif self.DataType == NumpyDataType:
            return self.Data[:, column] <= value
    
    def _perform_ne_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return self.Data[column] != value
        elif self.DataType == NumpyDataType:
            return self.Data[:, column] != value
    
    def _perform_nin_op(self, column, value):
        if self.DataType == DataFrameDataType:
            return ~self.Data[column].isin(value)
        elif self.DataType == NumpyDataType:
            raise Exception("Operation $nin must for NumpyDataType.")
        
    def _perform_isna_op(self, column, value):
        if self.DataType == DataFrameDataType:
            if value is True:
                return self.Data[column].isna()
            else:
                return self.Data[column].notna()
        elif self.DataType == NumpyDataType:
            raise Exception("Operation $isna must for NumpyDataType.")
    
    def _perform_comparison_operation(self, data, query) -> DataFrame:
        final_conditions_response = None
        resp = None
        for col, single_comparison_query in query.items():
            for operation, val in single_comparison_query.items():
                if self._is_eq_op(op_name=operation):
                    resp = self._perform_eq_op(column=col, value=val)
                elif self._is_gt_op(op_name=operation):
                    resp = self._perform_gt_op(column=col, value=val)
                elif self._is_gte_op(op_name=operation):
                    resp = self._perform_gte_op(column=col, value=val)
                elif self._is_in_op(op_name=operation):
                    resp = self._perform_in_op(column=col, value=val)
                elif self._is_lt_op(op_name=operation):
                    resp = self._perform_lt_op(column=col, value=val)
                elif self._is_lte_op(op_name=operation):
                    resp = self._perform_lte_op(column=col, value=val)
                elif self._is_ne_op(op_name=operation):
                    resp = self._perform_ne_op(column=col, value=val)
                elif self._is_nin_op(op_name=operation):
                    resp = self._perform_nin_op(column=col, value=val)
                elif self._is_isna_op(op_name=operation):
                    resp = self._perform_isna_op(column=col, value=val)
                else:
                    raise Exception(f"Invalid comparison operator: {single_comparison_query}")
                if final_conditions_response is None:
                    final_conditions_response = resp
                else:
                    final_conditions_response = self._perform_and_operation(
                        conditional1=final_conditions_response,
                        conditional2=resp
                    )
        return final_conditions_response
    
    
    ##########################################
    ### Logical Operators
    ##########################################
    def _is_and_syntax(self, query):
        if And in query:
            return True
        return False
    
    def _is_or_syntax(self, query):
        if Or in query:
            return True
        return False
        
    def _perform_and_operation(self, conditional1, conditional2):
        return conditional1 & conditional2
        
    def _perform_or_operation(self, conditional1, conditional2):
        return conditional1 | conditional2
    
    def _perform_logical_operation(self, conditional1: List, conditional2: List, query) -> DataFrame:
        resp = None
        if self._is_and_syntax(query=query):
            resp = self._perform_and_operation(conditional1=conditional1, conditional2=conditional2)
        elif self._is_or_syntax(query=query):
            resp = self._perform_or_operation(conditional1=conditional1, conditional2=conditional2)
        return resp
    
    ##########################################
    ### Find
    ##########################################
     
    def _find_on_numpy(self):
        def _recursive_query(query, depth: int=0):
            if self._is_valid_logical_syntax(query=query):
                final_query_conditional_response = None
                for k, v in query.items():
                    for expression in v:
                        conditional_response = _recursive_query(query=expression, depth=depth+1)
                        if final_query_conditional_response is None:
                            final_query_conditional_response = conditional_response
                        else:
                            final_query_conditional_response = self._perform_logical_operation(
                                conditional1=final_query_conditional_response,
                                conditional2=conditional_response,
                                query=query
                            )
                return final_query_conditional_response
            elif self._is_valid_comparison_syntax(query=query):
                real_query = {}
                for k, v in query.items():
                    real_query[int(self.DataColNameToIdx[k])] = v
                resp = self._perform_comparison_operation(
                    data=self.Data, 
                    query=real_query
                )
                return resp
        query_conditional = _recursive_query(query=self.Query, depth=0)
        return self.Data[query_conditional, :]
    
    def _find_on_dataframe(self):
        def _recursive_query(query, depth: int=0):
            if self._is_valid_logical_syntax(query=query):
                final_query_conditional_response = None
                for k, v in query.items():
                    for expression in v:
                        conditional_response = _recursive_query(query=expression, depth=depth+1)
                        if final_query_conditional_response is None:
                            final_query_conditional_response = conditional_response
                        else:
                            final_query_conditional_response = self._perform_logical_operation(
                                conditional1=final_query_conditional_response,
                                conditional2=conditional_response,
                                query=query
                            )
                return final_query_conditional_response
            elif self._is_valid_comparison_syntax(query=query):
                self.LOGGER.info(f'Comparison Query: {query}')
                resp = self._perform_comparison_operation(
                    data=self.Data, 
                    query=query
                )
                self.LOGGER.info(f'Valid rows: {resp.sum()} from {self.Data.shape[0]}')
                return resp
        query_conditional = _recursive_query(query=self.Query, depth=0)
        return self.Data[query_conditional]
    
    def run_query(self) -> Data:
        if self.DataType == NumpyDataType:
            return self._find_on_numpy()
        elif self.DataType == DataFrameDataType:
            return self._find_on_dataframe()


class TypeColumm(TypedDict):
    name: str
    typeraze: Callable


String = ['varchar', 'str']
Time = ['timestamp', 'datetime']
Int = ['int']
Float = ['float']
JSONB = ['JSONB', 'JSON']


def typeraze(
    dataframe: DataFrame,
    features: List, # List[FeatureJSON]
    dtypes: List[str]=None,
    ignore_dtypes: List[str]=[],
    custom_treatment: List[TypeColumm]=[],
    utc: Union[bool, None]=None,
    to_numeric: bool=False,
    LOGGER: Log=None,
    **kwargs
) -> DataFrame:
    """Type database columns.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame contining the data to be typed.
    features : List[SimpleFeature]
        The features to be typed.
    dtypes : List[str], optional
        Specify Dtypes to be handled, by default None
    ignore_dtypes : List[str], optional
        Ignore Dtypes when typing data, by default []
    custom_treatment : List[TypeColumm], optional
        If some colum needs to have a custom treatment 
        on data, the function to treat the column 
        can be specified here, by default []
    utc : Union[bool, None], optional
        If True, apply pd.to_datetime with utc=True, 
        if False, apply with False, by default None.
    to_numeric : bool, optional
        If True, apply pd.to_numeric before type, 
        if False, do not apply, by default False
    LOGGER : Log, optional
        The Log to verbose, by default None

    Returns
    -------
    DataFrame
        The typed DataFrame

    Raises
    ------
    Exception
        If Dtypes of column is not supported.
    Exception
        When some error occurs when typing some column.
    """
        
    df = dataframe.copy()

    AllDtypes = String + Time + Int + Float + JSONB
    
    if dtypes is None:
        dtypes = AllDtypes
        
    # filter features to process
    string_features_name = []
    time_features_name = []
    int_features_name = []
    float_features_name = []
    JSONB_features_name = []
    real_custom_treatment = []
    
    for feature in features:
        feature_type = feature['type']
        feature_name = feature['name']
        
        is_custom_treatment = False
        for custom_treat_col in custom_treatment:
            if feature_name == custom_treat_col['name']:
                real_custom_treatment.append(custom_treat_col)
                is_custom_treatment = True
                break            
        if is_custom_treatment:
            continue
        
        if feature_type in dtypes \
        and feature_type not in ignore_dtypes:
            
            if feature_type in String:
                string_features_name.append(feature_name)
                
            elif feature_type in Time:
                time_features_name.append(feature_name)
                
            elif feature_type in Int:
                int_features_name.append(feature_name)
                
            elif feature_type in Float:
                float_features_name.append(feature_name)
             
            elif feature_type in JSONB:
                JSONB_features_name.append(feature_name)
                
            elif feature_type not in AllDtypes:
                raise Exception(f'Invalid type {feature_type} on column {feature_name}')
        
    
    LOGGER.debug(f'Typing {len(string_features_name)} cols as String: {string_features_name}...')        
    archive_indexes = df[string_features_name].index
    temp_df = pd.DataFrame(
            np.where(
                pd.isnull(df[string_features_name]), 
                df[string_features_name], 
                df[string_features_name].astype(str)
            ),
            columns=string_features_name
        ).set_index(archive_indexes)
    for col in string_features_name:
        del df[col]
    df = pd.concat([df, temp_df], axis=1)
    LOGGER.debug(f'Typing {len(string_features_name)} cols as String... Done!')
        
    
    LOGGER.debug(f'Typing {len(time_features_name)} cols as Datetime: {time_features_name}...')       
    df[time_features_name] = df[time_features_name].apply(pd.to_datetime, errors='coerce', utc=utc)
    LOGGER.debug(f'Typing {len(time_features_name)} cols as Datetime... Done!')
        
    
    LOGGER.debug(f'Typing {len(int_features_name)} cols as Int64: {int_features_name}...') 
    if to_numeric:
        df[int_features_name] = df[int_features_name].apply(pd.to_numeric, errors='coerce')
    
    try:
        temp_df = df[int_features_name].astype('Int64')
    except:
        df[int_features_name] = df[int_features_name].apply(pd.to_numeric, errors='coerce')
        temp_df = df[int_features_name].astype('Int64')
        
    for col in int_features_name:
        del df[col]
    df = pd.concat([df, temp_df], axis=1)
    del temp_df
    LOGGER.debug(f'Typing {len(int_features_name)} cols as Int64... Done!')
        
        
    LOGGER.debug(f'Typing {len(float_features_name)} cols as Float: {float_features_name}...')
        
    if to_numeric:
        df[float_features_name] = df[float_features_name].apply(pd.to_numeric, errors='coerce')
        
    try:
        temp_df = df[float_features_name].astype(float)
    except:
        df[float_features_name] = df[float_features_name].apply(pd.to_numeric, errors='coerce')
        temp_df = df[float_features_name].astype(float)
        
    for col in float_features_name:
        del df[col]
    df = pd.concat([df, temp_df], axis=1)
    del temp_df
    LOGGER.debug(f'Typing {len(float_features_name)} cols as Float... Done!')
        
        
    LOGGER.debug(f'Typing {len(JSONB_features_name)} cols as JSONB: {JSONB_features_name}...') 
     
    for col in JSONB_features_name:
            
        LOGGER.debug(f'Typing col {col}...') 
                
        is_nan_indexes = df[df[col].isna()].index
        not_nan_indexes = df[df[col].notna()].index
        df_backup = df.copy()
        try:
            df[col] = df[col].astype(str)
            df.loc[df.index.isin(not_nan_indexes), col] \
                = df.loc[df.index.isin(not_nan_indexes), col] \
                    .apply(lambda x: ast.literal_eval(x))
        except:
            df = df_backup
            df.loc[df.index.isin(not_nan_indexes), col] \
                = df.loc[df.index.isin(not_nan_indexes), col] \
                    .apply(lambda x: ast.literal_eval(x) if type(x) == str else x)
            
        df.loc[df.index.isin(is_nan_indexes), col] = np.nan
            
        LOGGER.debug(f'Typing col {col}... Done!') 
        
    LOGGER.debug(f'Typing {len(JSONB_features_name)} cols as JSONB... Done!') 
    
    
    real_custom_treatment_names = [x['name'] for x in real_custom_treatment]
    LOGGER.debug(f'Handling custom treatment on {len(real_custom_treatment)} cols: {real_custom_treatment_names}...') 
    
    for custom_treat_col in real_custom_treatment:
        col = custom_treat_col['name']
        LOGGER.debug(f'Custom treatment on col {col}...') 
        custom_treat_typeraze = custom_treat_col['typeraze']
        df[col] = custom_treat_typeraze(df, col)
        LOGGER.debug(f'Custom treatment on col {col}... Done!') 
        
    LOGGER.debug(f'Handling custom treatment on {len(real_custom_treatment)} cols... Done!') 
    
    return df