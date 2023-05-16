""" 
This module was created to handle Features construction and 
other stuff related to features from Machine Learn Model.

"""


from drtools.utils import list_ops
from pandas import DataFrame
import pandas as pd
from typing import List, Union, Dict, TypedDict, Tuple
from drtools.utils import list_ops
from drtools.logs import Log, FormatterOptions
# from drtools.data_science.model_handling import Model
from drtools.data_science.general import typeraze
from enum import Enum


ColumnName = str
EncodeValue = List[Union[str, int]]
class EncondeOptions(TypedDict):
    EncodeValues: List[EncodeValue]
    DropRedundantColVal: str


def single_ohe(
    dataframe: DataFrame,
    column: str,
    encode_values: List[EncodeValue],
    prefix: str=None,
    prefix_sep: str="_",
    drop_self_col: bool=True,
    drop_redundant_col_val: str=None
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
    drop_redundant_col_val: str, optional
        If is not None, supply value that will corresnponde to encode column and 
        the encoded column will be dropped after generate encoded columns, 
        by default None.
        
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
        drop_cols = drop_cols + [column]
    df = df.drop(drop_cols, axis=1)
    # insert feature that not has on received dataframe
    for col in finals_ohe_cols:
        if col not in df.columns:
            df[col] = 0
    if drop_redundant_col_val is not None:
        drop_encoded_col_name = f'{prefix}{prefix_sep}{drop_redundant_col_val}'
        if drop_encoded_col_name in df.columns:
            df = df.drop(drop_encoded_col_name, axis=1)
    return df


def one_hot_encoding(
    dataframe: DataFrame,
    encode: Dict[ColumnName, EncondeOptions],
    prefix: str=None,
    prefix_sep: str="_",
    drop_self_col: bool=True,
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
    df = dataframe.copy()
    for column_name, encode_options in encode.items():
        encode_values = encode_options['EncodeValues']
        drop_redundant_col_val = encode_options.get('DropRedundantColVal', None)
        df = single_ohe(
            dataframe=df,
            column=column_name,
            encode_values=encode_values,
            drop_redundant_col_val=drop_redundant_col_val,
            prefix=prefix,
            prefix_sep=prefix_sep,
            drop_self_col=drop_self_col,
        )
    return df


class DataFrameMissingColumns(Exception):
    def __init__(
        self, 
        missing_cols: List[str], 
    ):
        self.missing_cols = missing_cols
        self.message = f"DataFrame has the following missing columns: {self.missing_cols}"
        super().__init__(self.message)
        
        
class DataFrameDiffLength(Exception):
    def __init__(
        self, 
        expected: int, 
        received: int, 
    ):
        self.expected = expected
        self.received = received
        self.message = f"DataFrames has different length. Expected: {self.expected} | Received: {self.received}"
        super().__init__(self.message)
        

class FeatureType(Enum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"


Input = 'input'
Output = 'output'
VarChar = 'varchar'
Str = 'str'
Int = 'int'
Float = 'float'
Datetime = 'datetime'
TimeStamp = 'timestamp'
Categorical = 'categorical'
Numerical = 'numerical'


class FeatureJSON(TypedDict):
    name: str
    type: Union[VarChar, Str, Int, Float, Datetime, TimeStamp]


class ExtendedFeatureJSON(FeatureJSON):
    description: Union[Categorical, Numerical]
    conditions: Dict
    observation: str


class Feature:
    def __init__(self, 
        name: str, 
        type: FeatureType,
        **kwargs,
    ) -> None:
        self.name = name
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @property
    def info(self) -> Dict:
        return self.__dict__


class Features:
    def __init__(self, features: List[Feature]=[]) -> None:
        self.features = features
        
    def list_features_name(self) -> List[str]:
        return [x.name for x in self.features]
    
    def append_features(self, features: List[Feature]) -> None:
        self.features = self.features + features
    
    @property
    def info(self) -> List[Dict]:
        return [feature.info for feature in self.features]
    

def construct_features(
    insert_features: Features,
    must_have_features: Features=Features(),
    type_must_have_features: bool=False,
    type_insert_features: bool=False,
    log_details: bool=True,
) -> DataFrame:
    """Decorator construct features and insert on DataFrame. The resulting 
    DataFrame must have same length from the received DataFrame.
    """
    def decorator(f):
        def wrapper(self, dataframe: DataFrame, *args, **kwargs):
            receveid_shape = dataframe.shape
            
            insert_features_name = insert_features.list_features_name()
            
            must_have_features_name = must_have_features.list_features_name()
            missing_cols = list_ops(must_have_features_name, dataframe)
            if len(missing_cols) > 0:
                raise DataFrameMissingColumns(missing_cols)
            
            if log_details:
                self.LOGGER.info(f'Constructing {insert_features_name} from {must_have_features_name}')
            
            if type_must_have_features:
                dataframe = typeraze(
                    dataframe, 
                    must_have_features.info, 
                    LOGGER=self.LOGGER
                )
            
            # execution
            response_dataframe = f(self, dataframe, *args, **kwargs)
            
            # insert_features_name = insert_features.list_features_name()
            missing_cols = list_ops(insert_features_name, response_dataframe)
            if len(missing_cols) > 0:
                raise DataFrameMissingColumns(missing_cols)
            
            if receveid_shape[0] != response_dataframe.shape[0]:
                raise DataFrameDiffLength(receveid_shape[0], response_dataframe.shape[0])
            
            if type_insert_features:
                response_dataframe = typeraze(
                    response_dataframe, 
                    insert_features.info, 
                    LOGGER=self.LOGGER
                )
            
            return response_dataframe
        wrapper.__doc__ = f.__doc__
        return wrapper
    return decorator


class BaseFeatureConstructor:
    def __init__(
        self, 
        name: str=None,
        model=None, # drtools.data_science.model_handling.Model
        LOGGER: Log=Log(
            name="FeatureConstructor",
            formatter_options=FormatterOptions(
                IncludeDate=True,
                IncludeLoggerName=True,
                IncludeLevelName=True,
                IncludeExecTime=False,
            ),
            default_start=False
        )
    ) -> None:
        self.name = name
        self.model = model
        self.LOGGER = LOGGER
        
    def set_model(
        self, 
        model # drtools.data_science.model_handling.Model
    ) -> None:
        self.model = model
        
    def set_logger(self, LOGGER: Log) -> None:
        self.LOGGER = LOGGER
        
    @construct_features
    def construct(self, dataframe: DataFrame, *args, **kwargs) -> DataFrame:
        pass


class BaseTransformer:
    def __init__(
        self,
        model, # drtools.data_science.model_handling.Model
        LOGGER: Log=Log(
            name="Transformer",
            formatter_options=FormatterOptions(
                IncludeDate=True,
                IncludeLoggerName=True,
                IncludeLevelName=True,
                IncludeExecTime=False,
            ),
            default_start=False
        )
    ) -> None:
        self.model = model
        self.LOGGER = LOGGER
    
    def apply(self, *args, **kwargs) -> Tuple:
        pass