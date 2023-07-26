""" 
This module was created to handle Features construction and 
other stuff related to features from Machine Learn Model.

"""


from drtools.utils import list_ops
from pandas import DataFrame, Series
import pandas as pd
from typing import List, Union, Dict, TypedDict, Any, Callable, Optional
from drtools.utils import list_ops
from drtools.logging import Logger, FormatterOptions
# from drtools.data_science.model_handling import Model
from drtools.data_science.general import typeraze
from enum import Enum
from copy import deepcopy


ColumnName = str
EncodeValue = List[Union[str, int]]
class EncondeOptions(TypedDict):
    EncodeValues: List[EncodeValue]
    DropRedundantColVal: str


def one_hot_encoding(
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
    STR = "str", "String"
    INT = "int", "Integer"
    FLOAT = "float", "Float"
    DATETIME = "datetime", "Datetime"
    DATETIMEUTC = "datetime[utc]", "Datetime UTC"
    TIMESTAMP = "timestamp", "Timestamp"
    JSONB = "JSONB", "JSONB"
    OBJECT = "object", "Object"
    BOOLEAN = "boolean", "Boolean"
    
    @property
    def name(self) -> str:
        return self.value[1]
    
    @property
    def code(self) -> str:
        return self.value[1]
    
    @classmethod
    def smart_instantiation(cls, value):
        upper_str_val = str(value).upper()
        obj = getattr(cls, upper_str_val, None)
        if obj is None:
            for feature_type in cls:
                if feature_type.code.upper() == upper_str_val:
                    obj = feature_type
                    break
        if obj is None:
            raise Exception(f"No correspondence was found for value: {value}")
        return obj


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


# class FeatureJSON(TypedDict):
#     name: str
#     type: Union[VarChar, Str, Int, Float, Datetime, TimeStamp]


# class ExtendedFeatureJSON(FeatureJSON):
#     description: Union[Categorical, Numerical]
#     conditions: Dict
#     observation: str


class Feature:
    def __init__(self, 
        name: str, 
        type: FeatureType=None,
        **kwargs,
    ) -> None:
        self.name = name
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @property
    def info(self) -> Dict:
        return {
            **self.__dict__,
            'name': self.name,
            'type': self.type.value if self.type is not None else None
        }


class Features:
    def __init__(
        self, 
        features: Optional[List[Feature]]=None
    ) -> None:
        if features is None:
            features = []
        self._features = features
        
    def list_features_name(self) -> List[str]:
        return [x.name for x in self._features]
    
    def append_features(self, features: List[Feature]) -> None:
        self._features = self._features + features
    
    def list_features(self) -> List[Feature]:
        return self._features
    
    def add_feature(self, feature: Feature):
        self._features.append(feature)
    
    @property
    def info(self) -> List[Dict]:
        return [feature.info for feature in self._features]


class BaseFeatureConstructor:
    
    def __init__(
        self, 
        features: Union[Features, Feature],
        must_have_features: Union[Features, Feature]=Features(),
        verbosity: bool=True,
        name: str=None,
        pre_validate: bool=True,
        post_validate: bool=True,
        spre_validate: bool=True,
        spost_validate: bool=True,
        constructor: Callable=None,
        sconstructor: Callable=None,
        model=None, # drtools.data_science.model_handling.Model
        LOGGER: Logger=Logger(
            name="BaseFeatureConstructor",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        self.features = features
        self.must_have_features = must_have_features
        self.verbosity = verbosity
        self.name = name
        self.pre_validate = pre_validate
        self.post_validate = post_validate
        self.spre_validate = spre_validate
        self.spost_validate = spost_validate
        
        if constructor is not None:
            self.constructor = constructor
            
        if sconstructor is not None:
            self.sconstructor = sconstructor
            
        self.model = model
        self.LOGGER = LOGGER
        self._startup()
    
    def _startup(self):
        self._original_features_is_Feature = None
        self._original_must_have_features_is_Feature = None
        
        if isinstance(self.features, Feature):
            self.features = Features([self.features])
            self._original_features_is_Feature = True
            
        elif isinstance(self.features, Features):
            self._original_features_is_Feature = False
            
        if isinstance(self.must_have_features, Feature):
            self.must_have_features = Features([self.must_have_features])
            self._original_must_have_features_is_Feature = True  
            
        elif isinstance(self.must_have_features, Features):
            self._original_must_have_features_is_Feature = False
    
    def set_model(
        self, 
        model # drtools.data_science.model_handling.Model
    ) -> None:
        self.model = model
        
    def set_logger(self, LOGGER: Logger) -> None:
        self.LOGGER = LOGGER
        
    def _get_features_name(self) -> List[str]:
        return self.features.list_features_name()  
        
    def _get_must_have_features_name(self) -> List[str]:
        return self.must_have_features.list_features_name()  
    
    def verbose(self, pre_validate: bool):
        features_name = self._get_features_name()
        must_have_features_name = self._get_must_have_features_name()
        
        if self.verbosity:
            
            if pre_validate:
                self.LOGGER.debug(f'Constructing {features_name} from {must_have_features_name}...')
                
            else:
                self.LOGGER.debug(f'Constructing {features_name} from {must_have_features_name}... Done!')        
    
    def _pre_validate(self, dataframe: DataFrame, **kwargs):
        must_have_features_name = self._get_must_have_features_name()
        missing_cols = list_ops(must_have_features_name, dataframe.columns)
        
        if self._original_must_have_features_is_Feature is None:
            raise Exception("Provided features parameter on BaseFeatureConstructor.__init__() must be Union[Features, Feature].")
        
        if len(missing_cols) > 0:
            raise DataFrameMissingColumns(missing_cols)
        
        # if self.type_must_have_features:
        #     dataframe = typeraze(dataframe, self.must_have_features.info, LOGGER=self.LOGGER)
            
        self.verbose(True)
            
    def _post_validate(self, response_dataframe: DataFrame, received_dataframe: DataFrame, **kwargs):
        receveid_shape = received_dataframe.shape
        features_name = self._get_features_name()
        missing_cols = list_ops(features_name, response_dataframe.columns)
        
        if len(missing_cols) > 0:
            raise DataFrameMissingColumns(missing_cols)
        
        if receveid_shape[0] != response_dataframe.shape[0]:
            raise DataFrameDiffLength(receveid_shape[0], response_dataframe.shape[0])
        
        # if self.type_features:
        #     response_dataframe = typeraze(response_dataframe, self.features.info, LOGGER=self.LOGGER)
            
        self.verbose(False)
    
    def _spre_validate(self, dataframe: DataFrame, **kwargs):
        must_have_features_name = self._get_must_have_features_name()
        missing_cols = list_ops(must_have_features_name, dataframe.columns)
        
        if self._original_features_is_Feature is not True:
            raise Exception("Provided features parameter on BaseFeatureConstructor.__init__() must be Feature.")
        
        if len(missing_cols) > 0:
            raise DataFrameMissingColumns(missing_cols)
        
        # if self.type_must_have_features:
        #     dataframe = typeraze(dataframe, self.must_have_features.info, LOGGER=self.LOGGER)
            
        self.verbose(True)
        
    def _spost_validate(
        self, 
        response_series: Series, 
        received_dataframe: DataFrame, 
        **kwargs
    ):
        self.verbose(False)
    
    def construct(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None,
        **kwargs
    ) -> DataFrame:
        
        if LOGGER is not None:
            self.set_logger(LOGGER)
        
        if self.pre_validate:
            self._pre_validate(dataframe, **kwargs)
            
        response_dataframe = self.constructor(dataframe, **kwargs)
        
        if self.post_validate:
            self._post_validate(response_dataframe, dataframe, **kwargs)
            
        return response_dataframe
    
    def sconstruct(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None,
        **kwargs
    ) -> Series:
        
        if LOGGER is not None:
            self.set_logger(LOGGER)
                
        if self.spre_validate:
            self._spre_validate(dataframe, **kwargs)
            
        responses_series = self.sconstructor(dataframe, **kwargs)
        
        if self.spost_validate:
            self._spost_validate(responses_series, dataframe, **kwargs)
            
        return responses_series
    
    def constructor(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        raise Exception("Must be implemented.")
    
    def sconstructor(self, dataframe: DataFrame, **kwargs) -> Series:
        raise Exception("Must be implemented.")


class BaseFeatureTyping(BaseFeatureConstructor):
    
    def __init__(
        self, 
        features: Union[Features, Feature],
        verbosity: bool=False,
        name: str=None,
        LOGGER: Logger=Logger(
            name="BaseFeatureTyping",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ):
        super(BaseFeatureTyping, self).__init__(
            features=features,
            must_have_features=deepcopy(features),
            verbosity=verbosity,
            name=name,
            LOGGER=LOGGER
        )
    
    def constructor(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return self.typing(dataframe, **kwargs)
    
    def sconstructor(self, dataframe: DataFrame, **kwargs) -> Series:
        series = dataframe[self._get_features_name()[0]]
        return self.styping(series, **kwargs)
    
    def type(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None,
        **kwargs
    ) -> DataFrame:
        return self.construct(dataframe, LOGGER=LOGGER, **kwargs)
    
    def stype(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None,
        **kwargs
    ) -> Series:
        return self.sconstruct(dataframe, LOGGER=LOGGER, **kwargs)
    
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        raise Exception("Must be implemented.")
    
    def styping(self, series: Series, **kwargs) -> Series:
        raise Exception("Must be implemented.")


class StringTyping(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        blank: bool = kwargs.pop("blank", True)
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .astype(pd.StringDtype())
        if not blank:
            dataframe[self._get_features_name()] \
                = dataframe[self._get_features_name()] \
                    .replace({"": None}) \
                    .astype(pd.StringDtype())
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        blank: bool = kwargs.pop("blank", True)
        series_response: Series = series.astype(pd.StringDtype())
        if not blank:
            series_response = series_response.astype(pd.StringDtype()) \
                .replace({"": None}) \
                .astype(pd.StringDtype())
        return series_response
    
    
class DatetimeUTCTyping(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .apply(pd.to_datetime, errors='coerce', utc=True)
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return pd.to_datetime(series, errors='coerce', utc=True)
    
    
class DatetimeTyping(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .apply(pd.to_datetime, errors='coerce')
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return pd.to_datetime(series, errors='coerce')
    
    
class Int64TypingLight(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .astype(pd.Int64Dtype())
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return series.astype(pd.Int64Dtype())


class Int64Typing(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .apply(pd.to_numeric, errors='coerce') \
                .astype(pd.Int64Dtype())
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return pd.to_numeric(series, errors='coerce').astype(pd.Int64Dtype())
    

class Int64TypingSmart(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        try:
            dataframe = Int64TypingLight(self.features).typing(dataframe, **kwargs)
        except Exception as exc:
            self.LOGGER.debug('Error typing using Light method, executing normal typing.')
        dataframe = Int64Typing(self.features).typing(dataframe, **kwargs)
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        try:
            series = Int64TypingLight(self.features).styping(series, **kwargs)
        except Exception as exc:
            self.LOGGER.debug('Error typing using Light method, executing normal typing.')
        series = Int64Typing(self.features).styping(series, **kwargs)
        return series    


class FloatTypingLight(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .astype('float')
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return series.astype('float')


class FloatTyping(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .apply(pd.to_numeric, errors='coerce') \
                .astype('float')
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return pd.to_numeric(series, errors='coerce').astype('float')
    

class FloatTypingSmart(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        try:
            dataframe = FloatTypingLight(self.features).typing(dataframe, **kwargs)
        except Exception as exc:
            self.LOGGER.debug('Error typing using Light method, executing normal typing.')
        dataframe = FloatTyping(self.features).typing(dataframe, **kwargs)
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        try:
            series = FloatTypingLight(self.features).styping(series, **kwargs)
        except Exception as exc:
            self.LOGGER.debug('Error typing using Light method, executing normal typing.')
        series = FloatTyping(self.features).styping(series, **kwargs)
        return series
    

class ObjectTyping(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .astype(object)
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return series.astype(object)
    

class BooleanTyping(BaseFeatureTyping):
    def typing(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe[self._get_features_name()] \
                .astype(pd.BooleanDtype())
        return dataframe
    
    def styping(self, series: Series, **kwargs) -> Series:
        return series.astype(pd.BooleanDtype())
    
    
class BaseMultiFeatureTyping:
    
    def __init__(
        self, 
        features: Features,
        LOGGER: Logger=Logger(
                name="BaseMultiFeatureTyping",
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            )
    ):
        self.features = features
        self.LOGGER = LOGGER
        self._feature_type_to_features: Dict[FeatureType, Features] \
            = self._group_features_by_typing_method()
        
    def set_logger(self, LOGGER: Logger) -> None:
        self.LOGGER = LOGGER
    
    def _pre_validate(self, dataframe: DataFrame):
        feature_names = self.features.list_features_name()
        diff_cols = list_ops(feature_names, dataframe.columns)
        if len(diff_cols) > 0:
            raise Exception(f"DataFrame does not have all necessary columns. Missing: {diff_cols}")
        
    def _post_validate(self, dataframe: DataFrame):
        pass
        
    def _group_features_by_typing_method(self) -> Dict[FeatureType, Features]:
        features: List[Feature] = self.features.list_features()
        feature_type_to_features = {}
        for feature in features:
            if feature.type not in feature_type_to_features:
                feature_type_to_features[feature.type] = Features()
            feature_type_to_features[feature.type].add_feature(feature)
        return feature_type_to_features
    
    @property
    def features_by_type(self) -> Dict[FeatureType, Features]:
        return self._feature_type_to_features
            
    def typing(self, dataframe: DataFrame) -> DataFrame:
        raise NotImplementedError
    
    def type(self, dataframe: DataFrame, LOGGER: Logger=None):
        if LOGGER is not None:
            self.set_logger(LOGGER)
        self._pre_validate(dataframe)
        response_dataframe = self.typing(dataframe)
        self._post_validate(response_dataframe)
        return response_dataframe


class FeaturesTyping(BaseMultiFeatureTyping):
    
    def typing(self, dataframe: DataFrame) -> DataFrame:
        
        for feature_type, features in self.features_by_type.items():
            
            if feature_type is FeatureType.STR:
                dataframe = StringTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.INT:
                dataframe = Int64Typing(features).type(dataframe)                
                
            elif feature_type is FeatureType.FLOAT:
                dataframe = FloatTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.DATETIME:
                dataframe = DatetimeTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.DATETIMEUTC:
                dataframe = DatetimeUTCTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.TIMESTAMP:
                dataframe = DatetimeTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.JSONB:
                dataframe = ObjectTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.OBJECT:
                dataframe = ObjectTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.BOOLEAN:
                dataframe = BooleanTyping(features).type(dataframe)
            
            else:
                raise Exception(f"Feature Type {feature_type} not supported.")
            
        return dataframe 


class LightFeaturesTyping(BaseMultiFeatureTyping):
    
    def typing(self, dataframe: DataFrame) -> DataFrame:
        
        for feature_type, features in self.features_by_type.items():
            
            if feature_type is FeatureType.STR:
                dataframe = StringTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.INT:
                dataframe = Int64TypingLight(features).type(dataframe)                
                
            elif feature_type is FeatureType.FLOAT:
                dataframe = FloatTypingLight(features).type(dataframe)
                
            elif feature_type is FeatureType.DATETIME:
                dataframe = DatetimeTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.DATETIMEUTC:
                dataframe = DatetimeUTCTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.TIMESTAMP:
                dataframe = DatetimeTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.JSONB:
                dataframe = ObjectTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.OBJECT:
                dataframe = ObjectTyping(features).type(dataframe)
                
            elif feature_type is FeatureType.BOOLEAN:
                dataframe = BooleanTyping(features).type(dataframe)
            
            else:
                raise Exception(f"Feature Type {feature_type} not supported.")
            
        return dataframe 


class SmartFeaturesTyping(BaseMultiFeatureTyping):
    
    def verbose(
        self, 
        feature_type: FeatureType, 
        features: Features, 
        done: bool=False
    ):
        verbose_message: str = f"Typing as {feature_type.name} the following features {features.list_features_name()}..."
        if done:
            verbose_message += " Done!"
        self.LOGGER.debug(verbose_message)
    
    def typing(self, dataframe: DataFrame) -> DataFrame:
        
        for feature_type, features in self.features_by_type.items():
            
            if feature_type is FeatureType.STR:
                self.verbose(feature_type, features)
                dataframe = StringTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.INT:
                self.verbose(feature_type, features)
                dataframe = Int64TypingSmart(features).type(dataframe)                
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.FLOAT:
                self.verbose(feature_type, features)
                dataframe = FloatTypingSmart(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.DATETIME:
                self.verbose(feature_type, features)
                dataframe = DatetimeTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.DATETIMEUTC:
                self.verbose(feature_type, features)
                dataframe = DatetimeUTCTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.TIMESTAMP:
                self.verbose(feature_type, features)
                dataframe = DatetimeTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.JSONB:
                self.verbose(feature_type, features)
                dataframe = ObjectTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.OBJECT:
                self.verbose(feature_type, features)
                dataframe = ObjectTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
                
            elif feature_type is FeatureType.BOOLEAN:
                self.verbose(feature_type, features)
                dataframe = BooleanTyping(features).type(dataframe)
                self.verbose(feature_type, features, done=True)
            
            else:
                raise Exception(f"Feature Type {feature_type} not supported.")
            
        return dataframe 


class BaseTransformer:
    def __init__(
        self,
        model, # drtools.data_science.model_handling.Model
        LOGGER: Logger=Logger(
            name="Transformer",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
                include_exec_time=False,
            ),
            default_start=False
        )
    ) -> None:
        self.model = model
        self.LOGGER = LOGGER
    
    def apply(self, *args, **kwargs) -> Any:
        pass