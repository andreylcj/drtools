""" 
This module was created to archive general functions, methods, 
classes and other stuff.

"""


from typing import Dict, List, Union, Optional, TypedDict
import numpy as np
from pandas import DataFrame
from drtools.utils import list_ops
from drtools.logs import Log


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