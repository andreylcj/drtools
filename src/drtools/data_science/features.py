""" 
This module was created to handle Features construction and 
other stuff related to features from Machine Learn Model.

"""


from pandas import DataFrame, Series
import pandas as pd
from typing import (
    List,
    Union,
    Dict,
    Any,
    Callable,
    Optional,
    Tuple,
)
from drtools.utils import list_ops
from drtools.logging import Logger, FormatterOptions
from enum import Enum
from copy import deepcopy
from datetime import timedelta


class OneHotEncoder:
    def __init__(
        self, 
        column: str, 
        encode_values: List[Union[str, int]], 
        prefix: str=None, 
        prefix_sep: str="_",
        drop_self_col: bool=True, 
        drop_redundant_col_val: str=None
    ):
        self.column = column
        self.encode_values = encode_values
        self.prefix = prefix
        self.prefix_sep = prefix_sep
        self.drop_self_col = drop_self_col
        self.drop_redundant_col_val = drop_redundant_col_val

    def encode(self, dataframe: DataFrame) -> DataFrame:
        """One hot encode one column, drop original column after 
        generate encoded and drop dummy cols that is not desired on 
        final data.
        
        Parameters
        ----------
        dataframe : DataFrame
            DataFrame containing data to encode.
            
        Returns
        -------
        DataFrame
            The DataFrame containing encoded columns.
        """
        if self.prefix is None:
            self.prefix = self.column    
        final_ohe_cols = [
            f'{self.prefix}{self.prefix_sep}{x}' 
            for x in self.encode_values
        ]
        df = dataframe.copy()
        dummies = pd.get_dummies(
            df[self.column], 
            prefix=self.prefix, 
            prefix_sep=self.prefix_sep
        )
        drop_cols = list_ops(dummies.columns, final_ohe_cols)
        df = pd.concat([df, dummies], axis=1)
        if self.drop_self_col:
            drop_cols.append(self.column)
        df = df.drop(drop_cols, axis=1)
        # insert feature that not has on received dataframe
        for col in final_ohe_cols:
            if col not in df.columns:
                df[col] = 0
        if self.drop_redundant_col_val is not None:
            drop_encoded_col_name = f'{self.prefix}{self.prefix_sep}{self.drop_redundant_col_val}'
            if drop_encoded_col_name in df.columns:
                df = df.drop(drop_encoded_col_name, axis=1)
        return df
    
    
class LabelEncoder:
    def __init__(
        self, 
        column: str, 
        values_mapper: Dict[str, int],
    ):
        self.column = column
        self.values_mapper = values_mapper

    def encode(self, dataframe: DataFrame) -> Series:
        str_values_mapper = {str(k): str(v) for k, v in self.values_mapper.items()}
        series = dataframe[self.column].replace(str_values_mapper)
        series = series.astype(int)
        return series


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
    STR = "string", "String", pd.StringDtype(), "O"
    
    INT8 = "int8", "Integer 8 bits", pd.Int8Dtype(), "int8"
    INT16 = "int16", "Integer 16 bits", pd.Int16Dtype(), "int16"
    INT32 = "int32", "Integer 32 bits", pd.Int32Dtype(), "int32"
    INT64 = "int64", "Integer 64 bits", pd.Int64Dtype(), "int64"
    
    UINT8 = "uint8", "Integer 8 bits", pd.UInt8Dtype(), "uint8"
    UINT16 = "uint16", "Integer 16 bits", pd.UInt16Dtype(), "uint16"
    UINT32 = "uint32", "Integer 32 bits", pd.UInt32Dtype(), "uint32"
    UINT64 = "uint64", "Integer 64 bits", pd.UInt64Dtype(), "uint64"
    
    FLOAT32 = "float32", "Float 32 bits", pd.Float32Dtype(), "float32"
    FLOAT64 = "float64", "Float 64 bits", pd.Float64Dtype(), "float64"
    
    DATETIME = "datetime64[ns]", "Datetime", None, None
    DATETIMEUTC = "datetime64[ns, UTC]", "Datetime UTC", None, None
    TIMESTAMP = "timestamp", "Timestamp", None, None
    
    JSONB = "JSONB", "JSONB", object, "O"
    OBJECT = "object", "Object", object, "O"
    BOOLEAN = "boolean", "Boolean", pd.BooleanDtype(), "O"
    
    @property
    def code(self) -> str:
        return self.value[0]
    
    @property
    def pname(self) -> str:
        return self.value[1]
    
    def type(self, numpy: bool=False):
        return self.value[3] if numpy else self.value[2]
    
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


class Feature:
    def __init__(
        self, 
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
            'type': self.type.code if self.type is not None else None
        }
        
    def __eq__(self, other): 
        if not isinstance(other, Feature):
            return False
        return self.name == other.name and self.type is other.type


class StringFeature(Feature):
    def __init__(
        self, 
        name: str,
        blank: bool=True,
        **kwargs
    ) -> None:
        super(StringFeature, self).__init__(
            name=name,
            type=FeatureType.STR,
            blank=blank,
            **kwargs
        )
        
        
class Int8Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(Int8Feature, self).__init__(
            name=name, 
            type=FeatureType.INT8,
            **kwargs
        )


class Int16Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(Int16Feature, self).__init__(
            name=name, 
            type=FeatureType.INT16,
            **kwargs
        )


class Int32Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(Int32Feature, self).__init__(
            name=name, 
            type=FeatureType.INT32,
            **kwargs
        )


class Int64Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(Int64Feature, self).__init__(
            name=name, 
            type=FeatureType.INT64,
            **kwargs
        )


class UInt8Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(UInt8Feature, self).__init__(
            name=name, 
            type=FeatureType.UINT8,
            **kwargs
        )


class UInt16Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(UInt16Feature, self).__init__(
            name=name, 
            type=FeatureType.UINT16,
            **kwargs
        )


class UInt32Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(UInt32Feature, self).__init__(
            name=name, 
            type=FeatureType.UINT32,
            **kwargs
        )


class UInt64Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(UInt64Feature, self).__init__(
            name=name, 
            type=FeatureType.UINT64,
            **kwargs
        )


class Float32Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(Float32Feature, self).__init__(
            name=name, 
            type=FeatureType.FLOAT32,
            **kwargs
        )


class Float64Feature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(Float64Feature, self).__init__(
            name=name, 
            type=FeatureType.FLOAT64,
            **kwargs
        )


class DatetimeFeature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(DatetimeFeature, self).__init__(
            name=name, 
            type=FeatureType.DATETIME,
            **kwargs
        )


class DatetimeUtcFeature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(DatetimeUtcFeature, self).__init__(
            name=name, 
            type=FeatureType.DATETIMEUTC,
            **kwargs
        )


class TimestampFeature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(TimestampFeature, self).__init__(
            name=name, 
            type=FeatureType.TIMESTAMP,
            **kwargs
        )


class JsonbFeature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(JsonbFeature, self).__init__(
            name=name, 
            type=FeatureType.JSONB,
            **kwargs
        )


class ObjectFeature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(ObjectFeature, self).__init__(
            name=name, 
            type=FeatureType.OBJECT,
            **kwargs
        )


class BooleanFeature(Feature):
    def __init__(
        self, 
        name: str,
        **kwargs
    ) -> None:
        super(BooleanFeature, self).__init__(
            name=name, 
            type=FeatureType.BOOLEAN,
            **kwargs
        )


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
        return [feature for feature in self._features]
    
    def add_feature(self, feature: Feature):
        self._features.append(feature)
    
    def get_feature_by_name(self, name: str) -> Optional[Feature]:
        for feature in self._features:
            if feature.name == name:
                return self._features
        return None
    
    def remove_feature(self, feature: Feature) -> Feature:
        for idx, _feature in enumerate(self._features):
            if feature == _feature:
                break
        return self._features.pop(idx)
    
    def remove_feature_by_name(self, name: str) -> Feature:
        for feature in self._features:
            if feature.name == name:
                break
        return self.remove_feature(feature)
    
    def remove_features(self, features):
        removed_features: List[Feature] = []
        for feature2 in features:
            for feature1 in self._features:
                if feature2 == feature1:
                    removed_features.append(self.remove_feature(feature2))
        return Features(removed_features)
    
    def drop_duplicates(self):
        final_features: List[Feature] = []
        for feature in self._features:
            drop: bool = False
            for feature1 in final_features:
                if feature == feature1:
                    drop = True
                    break
            if not drop:
                final_features.append(feature)
        self._features = Features(final_features)
        return self._features
    
    @property
    def features(self) -> List[Feature]:
        return [feature for feature in self._features]
    
    @property
    def features_name(self) -> List[str]:
        return [feature.name for feature in self._features]
    
    @property
    def info(self) -> List[Dict]:
        return [feature.info for feature in self._features]
    
    def __iter__(self):
        return iter(list(self._features))
    
    def __len__(self):
        return len(self._features)
    
    def __getitem__(self, item: int):
        return self._features[item]
    
    def __add__(self, other):
        return Features(self._features + other._features)


class BaseFeatureConstructor:
    
    def __init__(
        self, 
        features: Union[Features, Feature],
        must_have_features: Union[Features, Feature]=Features(),
        verbosity: bool=True,
        name: str=None,
        type_features: bool=False,
        type_must_have_features: bool=False,
        pre_validate: bool=True,
        post_validate: bool=True,
        spre_validate: bool=True,
        spost_validate: bool=True,
        constructor: Callable=None,
        sconstructor: Callable=None,
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
        self.type_features = type_features
        self.type_must_have_features = type_must_have_features
        self.pre_validate = pre_validate
        self.post_validate = post_validate
        self.spre_validate = spre_validate
        self.spost_validate = spost_validate
        
        if constructor is not None:
            self.constructor = constructor
            
        if sconstructor is not None:
            self.sconstructor = sconstructor
            
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
    
    def _pre_validate(
        self, 
        dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        must_have_features_name = self._get_must_have_features_name()
        missing_cols = list_ops(must_have_features_name, dataframe.columns)
        
        if self._original_must_have_features_is_Feature is None:
            raise Exception("Provided features parameter on BaseFeatureConstructor.__init__() must be Union[Features, Feature].")
        
        if len(missing_cols) > 0:
            raise DataFrameMissingColumns(missing_cols)
        
        self.verbose(True)
        
        return dataframe
            
    def _post_validate(
        self, 
        response_dataframe: DataFrame, 
        received_dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        receveid_shape: Tuple[int, int] = received_dataframe.shape
        features_name: List[str] = self._get_features_name()
        missing_cols: List[str] = list_ops(features_name, response_dataframe.columns)
        not_expected_cols: List[str] = list_ops(
            response_dataframe.columns, 
            list_ops(received_dataframe.columns, features_name, ops='union')
        )
        
        if len(missing_cols) > 0:
            raise DataFrameMissingColumns(missing_cols)
        
        if len(not_expected_cols) > 0:
            raise Exception(f"Not expected cols: {not_expected_cols}")
        
        # if receveid_shape[0] != response_dataframe.shape[0]:
        #     raise DataFrameDiffLength(receveid_shape[0], response_dataframe.shape[0])
        
        self.verbose(False)
        
        return response_dataframe
    
    def _spre_validate(
        self, 
        dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        must_have_features_name = self._get_must_have_features_name()
        missing_cols = list_ops(must_have_features_name, dataframe.columns)
        
        if self._original_features_is_Feature is not True:
            raise Exception("Provided features parameter on BaseFeatureConstructor.__init__() must be Feature.")
        
        if len(missing_cols) > 0:
            raise DataFrameMissingColumns(missing_cols)
        
        self.verbose(True)
        
        return dataframe
        
    def _spost_validate(
        self, 
        response_series: Series, 
        received_dataframe: DataFrame, 
        **kwargs
    ) -> Series:
        self.verbose(False)
        return response_series
    
    def construct(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None,
        **kwargs
    ) -> DataFrame:
        
        if LOGGER is not None:
            self.set_logger(LOGGER)
        
        if self.pre_validate:
            dataframe = self._pre_validate(dataframe, **kwargs)
        
        response_dataframe = self.constructor(dataframe, **kwargs)
        
        if self.post_validate:
            response_dataframe = self._post_validate(response_dataframe, dataframe, **kwargs)
            
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
            dataframe = self._spre_validate(dataframe, **kwargs)
            
        responses_series = self.sconstructor(dataframe, **kwargs)
        
        if self.spost_validate:
            responses_series = self._spost_validate(responses_series, dataframe, **kwargs)
            
        return responses_series
    
    def constructor(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        raise Exception("Must be implemented.")
    
    def sconstructor(self, dataframe: DataFrame, **kwargs) -> Series:
        raise Exception("Must be implemented.")


class BaseFeatureTyper(BaseFeatureConstructor):
    
    def __init__(
        self, 
        features: Union[Features, Feature],
        verbosity: bool=False,
        name: str=None,
        numpy: bool=False,
        LOGGER: Logger=Logger(
            name="BaseFeatureTyper",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ):
        super(BaseFeatureTyper, self).__init__(
            features=features,
            must_have_features=deepcopy(features),
            verbosity=verbosity,
            name=name,
            LOGGER=LOGGER
        )
        self.numpy = numpy
    
    def constructor(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return self.typer(dataframe, **kwargs)
    
    def sconstructor(self, dataframe: DataFrame, **kwargs) -> Series:
        series = dataframe.loc[:, self._get_features_name()[0]]
        return self.styper(series, **kwargs)
    
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
    
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        raise Exception("Must be implemented.")
    
    def styper(self, series: Series, **kwargs) -> Series:
        raise Exception("Must be implemented.")


#################################
# String Typer
#################################
class StringTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .astype(FeatureType.STR.type(self.numpy))
        
        # features: List[StringFeature] = self.features.list_features()
        blank_features: Features = Features([
            feature
            for feature in self.features
            if not getattr(feature, 'blank', True)
        ])
        
        if len(blank_features) > 0:
            dataframe[blank_features.list_features_name()  ] \
                = dataframe.loc[:, blank_features.list_features_name()  ] \
                    .replace({"": None}) \
                    .astype(FeatureType.STR.type(self.numpy))

        return dataframe
    
    def styper(self, series: Series, **kwargs) -> Series:
        series_response: Series = series.astype(FeatureType.STR.type(self.numpy))
        feature: StringFeature = self.features[0]
        blank: bool = getattr(feature, 'blank', True)
        if not blank:
            series_response = series_response \
                .replace({"": None}) \
                .astype(FeatureType.STR.type(self.numpy))
        return series_response
    

#################################
# Datetime Typer
#################################
class DatetimeUtcTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .apply(pd.to_datetime, errors='coerce', utc=True)
        return dataframe
    
    def styper(self, series: Series, **kwargs) -> Series:
        return pd.to_datetime(series, errors='coerce', utc=True)
    
    
class DatetimeTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .apply(pd.to_datetime, errors='coerce')
        return dataframe
    
    def styper(self, series: Series, **kwargs) -> Series:
        return pd.to_datetime(series, errors='coerce')
    


#################################
# Number Typer
#################################
class NumberLightTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, type, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .astype(type)
        return dataframe
    
    def styper(self, series: Series, type, **kwargs) -> Series:
        return series.astype(type)
    
    
class NumberTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, type, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .apply(pd.to_numeric, errors='coerce') \
                .astype(type)
        return dataframe
    
    def styper(self, series: Series, type, **kwargs) -> Series:
        return pd.to_numeric(series, errors='coerce') \
                .astype(type)
    
    
class SmartNumberTyper(BaseFeatureTyper):
    
    def smart_get_type(self, series: Series, **kwargs):
        raise NotImplementedError
    
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .apply(pd.to_numeric, errors='coerce')
                
        for col_name in self._get_features_name():
            smart_type = self.smart_get_type(dataframe[col_name], **kwargs)
            dataframe[col_name] = dataframe[col_name].astype(smart_type)
        
        return dataframe
    
    def styper(self, series: Series, **kwargs) -> Series:
        series = pd.to_numeric(series, errors='coerce')
        smart_type = self.smart_get_type(series, **kwargs)
        series = series.astype(smart_type)
        return series


#################################
# Int Typer
#################################
class Int8Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(Int8Typer, self).typer(dataframe, FeatureType.INT8.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(Int8Typer, self).styper(series, FeatureType.INT8.type(self.numpy), **kwargs)


class Int16Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(Int16Typer, self).typer(dataframe, FeatureType.INT16.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(Int16Typer, self).styper(series, FeatureType.INT16.type(self.numpy), **kwargs)


class Int32Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(Int32Typer, self).typer(dataframe, FeatureType.INT32.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(Int32Typer, self).styper(series, FeatureType.INT32.type(self.numpy), **kwargs)


class Int64Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(Int64Typer, self).typer(dataframe, FeatureType.INT64.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(Int64Typer, self).styper(series, FeatureType.INT64.type(self.numpy), **kwargs)


class UInt8Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(UInt8Typer, self).typer(dataframe, FeatureType.UINT8.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(UInt8Typer, self).styper(series, FeatureType.UINT8.type(self.numpy), **kwargs)


class UInt16Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(UInt16Typer, self).typer(dataframe, FeatureType.UINT16.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(UInt16Typer, self).styper(series, FeatureType.UINT16.type(self.numpy), **kwargs)


class UInt32Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(UInt32Typer, self).typer(dataframe, FeatureType.UINT32.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(UInt32Typer, self).styper(series, FeatureType.UINT32.type(self.numpy), **kwargs)


class UInt64Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(UInt64Typer, self).typer(dataframe, FeatureType.UINT64.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(UInt64Typer, self).styper(series, FeatureType.UINT64.type(self.numpy), **kwargs)
    

class SmartIntTyper(SmartNumberTyper):
    
    def smart_get_type(self, series: Series, **kwargs):
        max_val = series.max()
        min_val = series.min()
        col_smart_type = None
        
        if -128 <= min_val and max_val <= 127:
            col_smart_type = FeatureType.INT8.type(self.numpy)
            
        elif 0 <= min_val and max_val <= 255:
            col_smart_type = FeatureType.UINT8.type(self.numpy)
            
        elif -32768 <= min_val and max_val <= 32767:
            col_smart_type = FeatureType.INT16.type(self.numpy)
            
        elif 0 <= min_val and max_val <= 65535:
            col_smart_type = FeatureType.UINT16.type(self.numpy)
            
        elif -2147483648 <= min_val and max_val <= 2147483647:
            col_smart_type = FeatureType.INT32.type(self.numpy)
            
        elif 0 <= min_val and max_val <= 4294967295:
            col_smart_type = FeatureType.UINT32.type(self.numpy)
            
        elif -9223372036854775808 <= min_val and max_val <= 9223372036854775807:
            col_smart_type = FeatureType.INT64.type(self.numpy)
            
        elif 0 <= min_val:
            col_smart_type = FeatureType.UINT64.type(self.numpy)
            
        else:
            raise Exception(f"Unsupported column value. Min: {min_val} | Max: {max_val}")
            
        return col_smart_type


#################################
# Float Typer
#################################
class Float32Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(Float32Typer, self).typer(dataframe, FeatureType.FLOAT32.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(Float32Typer, self).styper(series, FeatureType.FLOAT32.type(self.numpy), **kwargs)


class Float64Typer(NumberTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        return super(Float64Typer, self).typer(dataframe, FeatureType.FLOAT64.type(self.numpy), **kwargs)
    
    def styper(self, series: Series, **kwargs) -> Series:
        return super(Float64Typer, self).styper(series, FeatureType.FLOAT64.type(self.numpy), **kwargs)
    

class SmartFloatTyper(SmartNumberTyper):
    
    def smart_get_type(self, series: Series, **kwargs):
        max_val = abs(series.max())
        col_smart_type = None
        
        if max_val <= 3.4028235e+38:
            col_smart_type = FeatureType.FLOAT32.type(self.numpy)
            
        else: # 1.7976931348623157e+308
            col_smart_type = FeatureType.FLOAT64.type(self.numpy)
            
        return col_smart_type
    

#################################
# JSONB & Object Typer
#################################
class ObjectTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .astype(FeatureType.OBJECT.type(self.numpy))
        return dataframe
    
    def styper(self, series: Series, **kwargs) -> Series:
        return series.astype(FeatureType.OBJECT.type(self.numpy))
    

#################################
# Boolean Typer
#################################
class BooleanTyper(BaseFeatureTyper):
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe[self._get_features_name()] \
            = dataframe.loc[:, self._get_features_name()] \
                .astype(FeatureType.BOOLEAN.type(self.numpy))
        return dataframe
    
    def styper(self, series: Series, **kwargs) -> Series:
        return series.astype(FeatureType.BOOLEAN.type(self.numpy))
    
    
class BaseMultiFeatureTyper:
    
    def __init__(
        self, 
        features: Union[Feature, Features],
        verbosity: bool=False,
        numpy: bool=False,
        LOGGER: Logger=Logger(
                name="BaseMultiFeatureTyper",
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            )
    ):           
        self.features = features
        self.verbosity = verbosity
        self.numpy = numpy
        self.LOGGER = LOGGER
        self._startup()
            
    def _startup(self):            
        self._original_features_is_Feature = None
        
        if isinstance(self.features, Feature):
            self.features = Features([self.features])
            self._original_features_is_Feature = True
            
        elif isinstance(self.features, Features):
            self._original_features_is_Feature = False
            
        self._feature_type_to_features: Dict[FeatureType, Features] \
            = self._group_features_by_typing_method()
        
    def set_logger(self, LOGGER: Logger) -> None:
        self.LOGGER = LOGGER
    
    def _pre_validate(self, received_dataframe: DataFrame, **kwargs):
        feature_names = self.features.list_features_name()
        diff_cols = list_ops(feature_names, received_dataframe.columns)
        if len(diff_cols) > 0:
            raise Exception(f"DataFrame does not have all necessary columns. Missing: {diff_cols}")
        
    def _post_validate(self, response_dataframe: DataFrame, received_dataframe: DataFrame, **kwargs):
        pass
    
    def _spre_validate(self, received_dataframe: DataFrame, **kwargs):    
        if self._original_features_is_Feature is not True:
            raise Exception("Provided features parameter on BaseMultiFeatureTyper.__init__() must be Feature.")        
        self._pre_validate(received_dataframe)
        
    def _spost_validate(
        self, 
        response_series: Series, 
        received_dataframe: DataFrame, 
        **kwargs
    ):
        pass
        
    def _group_features_by_typing_method(self) -> Dict[FeatureType, Features]:
        # features: List[Feature] = self.features.list_features()
        feature_type_to_features = {}
        for feature in self.features:
            if feature.type not in feature_type_to_features:
                feature_type_to_features[feature.type] = Features()
            feature_type_to_features[feature.type].add_feature(feature)
        return feature_type_to_features
    
    @property
    def features_by_type(self) -> Dict[FeatureType, Features]:
        return self._feature_type_to_features
    
    def verbose_typing(
        self, 
        feature_type: FeatureType, 
        features: Features, 
        done: bool=False
    ):
        verbose_message: str = f"Typing as {feature_type.code} the following features {features.list_features_name()}..."
        if done:
            verbose_message += " Done!"
            
        if self.verbosity:
            self.LOGGER.debug(verbose_message)
    
    def verbose_typer(
        self, 
        dataframe: DataFrame,
        features: Features,
        typer: BaseFeatureTyper,
        feature_type: FeatureType,
        **kwargs
    ) -> DataFrame:
        self.verbose_typing(feature_type, features)
        dataframe = typer(features, numpy=self.numpy).type(dataframe, **kwargs)
        self.verbose_typing(feature_type, features, done=True)
        return dataframe
    
    def type(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None, 
        verbosity: bool=True,
        **kwargs
    ) -> DataFrame:
        self.verbosity = verbosity
        if LOGGER is not None:
            self.set_logger(LOGGER)
        self._pre_validate(dataframe, **kwargs)
        
        if dataframe.shape[0] > 0:
            response_dataframe = self.typer(dataframe, **kwargs)
            
        else:
            response_dataframe = dataframe
            
        self._post_validate(response_dataframe, dataframe, **kwargs)
        return response_dataframe
    
    def stype(
        self, 
        dataframe: DataFrame, 
        LOGGER: Logger=None, 
        verbosity: bool=None,
        **kwargs
    ) -> Series:
        
        if verbosity is not None:
            self.verbosity = verbosity
            
        if LOGGER is not None:
            self.set_logger(LOGGER)
            
        self._spre_validate(dataframe, **kwargs)
        
        if dataframe.shape[0] > 0:
            response_series = self.styper(dataframe, **kwargs)
            
        else:
            feature_name: str = self.features.list_features_name()[0]
            response_series = dataframe.loc[:, feature_name]
        
        self._spost_validate(response_series, dataframe, **kwargs)
        return response_series
            
    def typer(self, dataframe: DataFrame) -> DataFrame:
        raise NotImplementedError
            
    def styper(self, dataframe: DataFrame) -> Series:
        raise NotImplementedError


class FeaturesTyper(BaseMultiFeatureTyper):
    
    def styper(self, dataframe: DataFrame, **kwargs) -> Series:
        feature_name: str = self.features.list_features_name()[0]
        single_col_df = dataframe.loc[:, [feature_name]]
        response_dataframe: DataFrame = self.typer(single_col_df, **kwargs)
        return response_dataframe.loc[:, feature_name]
    
    def typer(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        
        for feature_type, features in self.features_by_type.items():
            
            # String
            if feature_type is FeatureType.STR:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=StringTyper,
                    feature_type=feature_type,
                    **kwargs
                )
                
            # Int
            elif feature_type is FeatureType.INT8:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=Int8Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.INT16:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=Int16Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.INT32:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=Int32Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.INT64:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=Int64Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.UINT8:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=UInt8Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.UINT16:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=UInt16Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.UINT32:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=UInt32Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.UINT64:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=UInt64Typer,
                    feature_type=feature_type,
                    **kwargs
                )
            
            # Float                
            elif feature_type is FeatureType.FLOAT32:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=Float32Typer,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.FLOAT64:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=Float64Typer,
                    feature_type=feature_type,
                    **kwargs
                )
            
            # Datetime
            elif feature_type is FeatureType.DATETIME:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=DatetimeTyper,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.DATETIMEUTC:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=DatetimeUtcTyper,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.TIMESTAMP:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=DatetimeTyper,
                    feature_type=feature_type,
                    **kwargs
                )
            
            # Object
            elif feature_type is FeatureType.JSONB:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=ObjectTyper,
                    feature_type=feature_type,
                    **kwargs
                )
                
            elif feature_type is FeatureType.OBJECT:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=ObjectTyper,
                    feature_type=feature_type,
                    **kwargs
                )
            
            # Boolean
            elif feature_type is FeatureType.BOOLEAN:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=BooleanTyper,
                    feature_type=feature_type,
                    **kwargs
                )
            
            else:
                raise Exception(f"Feature Type {feature_type} not supported.")
            
        return dataframe


class SmartFeaturesTyper(FeaturesTyper):
    
    def typer(self, dataframe: DataFrame) -> DataFrame:
        
        for feature_type, features in self.features_by_type.items():
            
            # String
            if feature_type is FeatureType.STR:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=StringTyper,
                    feature_type=feature_type
                )
                
            # Int
            elif 'int' in feature_type.code:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=SmartIntTyper,
                    feature_type=feature_type
                )
            
            # Float
            elif 'float' in feature_type.code:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=SmartFloatTyper,
                    feature_type=feature_type
                )
            
            # Datetime
            elif feature_type is FeatureType.DATETIME:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=DatetimeTyper,
                    feature_type=feature_type
                )
                
            elif feature_type is FeatureType.DATETIMEUTC:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=DatetimeUtcTyper,
                    feature_type=feature_type
                )
                
            elif feature_type is FeatureType.TIMESTAMP:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=DatetimeTyper,
                    feature_type=feature_type
                )
            
            # Object
            elif feature_type is FeatureType.JSONB:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=ObjectTyper,
                    feature_type=feature_type
                )
                
            elif feature_type is FeatureType.OBJECT:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=ObjectTyper,
                    feature_type=feature_type
                )
            
            # Boolean
            elif feature_type is FeatureType.BOOLEAN:
                dataframe = self.verbose_typer(
                    dataframe=dataframe,
                    features=features,
                    typer=BooleanTyper,
                    feature_type=feature_type
                )
            
            else:
                raise Exception(f"Feature Type {feature_type} not supported.")
            
        return dataframe


class BaseTyperFeatureConstructor(BaseFeatureConstructor):
    
    def _pre_validate(
        self, 
        dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        dataframe = super(BaseTyperFeatureConstructor, self)._pre_validate(
            dataframe=dataframe,
            **kwargs
        )
        
        if self.type_must_have_features:
            dataframe = FeaturesTyper( 
                self.must_have_features, 
                LOGGER=self.LOGGER
            ).type(
                dataframe=dataframe,
                **kwargs
            )
        
        return dataframe
            
    def _post_validate(
        self, 
        response_dataframe: DataFrame, 
        received_dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        dataframe = super(BaseTyperFeatureConstructor, self)._post_validate(
            response_dataframe=response_dataframe,
            received_dataframe=received_dataframe,
            **kwargs
        )
        
        if self.type_features:
            dataframe = FeaturesTyper( 
                self.features, 
                LOGGER=self.LOGGER
            ).type(
                dataframe=response_dataframe,
                **kwargs
            )
        
        return dataframe
    
    def _spre_validate(
        self, 
        dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        dataframe = super(BaseTyperFeatureConstructor, self)._spre_validate(
            dataframe=dataframe,
            **kwargs
        )
        
        if self.type_must_have_features:
            dataframe = FeaturesTyper( 
                self.must_have_features, 
                LOGGER=self.LOGGER
            ).type(
                dataframe=dataframe,
                **kwargs
            )
        
        return dataframe
        
    def _spost_validate(
        self, 
        response_series: Series, 
        received_dataframe: DataFrame, 
        **kwargs
    ) -> Series:
        response_series = super(BaseTyperFeatureConstructor, self)._spost_validate(
            response_series=response_series,
            received_dataframe=received_dataframe,
            **kwargs
        )
        
        return response_series
    
    
class BaseErrorTolerance:
    def __init__(
        self,
        feature: Feature,
        value: Any,
    ):
        self.feature = feature
        self.value = value


class NumericErrorTolerance(BaseErrorTolerance):
    def __init__(
        self,
        feature: Feature,
        value: float,
    ):
        super(NumericErrorTolerance, self).__init__(
            feature,
            value
        )


class DatetimeErrorTolerance(BaseErrorTolerance):
    def __init__(
        self,
        feature: Feature,
        value: timedelta,
    ):
        super(DatetimeErrorTolerance, self).__init__(
            feature,
            value
        )


class BaseCustomErrorTolerance:
    def __init__(
        self,
        error_tolerances: List[BaseErrorTolerance]=None,
        default_tolerance: float=None,
    ):
        if error_tolerances is None:
            error_tolerances = []
        self.error_tolerances = error_tolerances
        self.default_tolerance = default_tolerance
        
    def get_tolerance_by_feature(self, feature: Feature) -> Any:
        for error_tolerance in self.error_tolerances:
            if error_tolerance.feature == feature:
                return error_tolerance.value
        return self.default_tolerance


class NumericCustomErrorTolerance(BaseCustomErrorTolerance):
    def __init__(
        self,
        error_tolerances: List[NumericErrorTolerance]=None,
        default_tolerance: float=1e-3,
    ):
        super(NumericCustomErrorTolerance, self).__init__(
            error_tolerances,
            default_tolerance
        )


class DatetimeCustomErrorTolerance(BaseCustomErrorTolerance):
    def __init__(
        self,
        error_tolerances: List[DatetimeErrorTolerance]=None,
        default_tolerance: timedelta=timedelta(seconds=1),
    ):
        super(DatetimeCustomErrorTolerance, self).__init__(
            error_tolerances,
            default_tolerance
        )


class BaseFeaturesValidator:
    
    def __init__(
        self,
        features: Features,
        merge_on: Features,
        constructor: Optional[BaseFeatureConstructor]=None,
        error_log_level: int=1, # 1 or 2
        numeric_error_tolerance: NumericCustomErrorTolerance=NumericCustomErrorTolerance(),
        datetime_error_tolerance: DatetimeCustomErrorTolerance=DatetimeCustomErrorTolerance(),
        LOGGER: Logger=Logger(
                name="BaseFeaturesValidator",
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            )
    ) -> None:
        self.constructor = constructor
        self.features = deepcopy(features)
        self.merge_on = deepcopy(merge_on)
        self.unique_features_removed_merge_on = deepcopy(features)
        self.unique_features_removed_merge_on.remove_features(self.merge_on)
        self._full_features_name: List[str] \
            = self.merge_on.list_features_name() \
                + self.unique_features_removed_merge_on.list_features_name()
        self.error_log_level = error_log_level
        self.numeric_error_tolerance = numeric_error_tolerance
        self.datetime_error_tolerance = datetime_error_tolerance
        self.LOGGER = LOGGER
    
    def _validate(
        self,
        expected_df: DataFrame,
        received_df: DataFrame,
    ):        
        received_df = received_df.rename({
                col: f'received.{col}'
                for col in received_df.columns
            },
            axis=1
        )
        
        expected_df = expected_df.rename({
                col: f'expected.{col}'
                for col in expected_df.columns
            },
            axis=1
        )
        
        right_on = [
            f'received.{merge_feature_name}'
            for merge_feature_name in self.merge_on.list_features_name()
        ]
        left_on = [
            f'expected.{merge_feature_name}'
            for merge_feature_name in self.merge_on.list_features_name()
        ]
        
        merged_df = expected_df.merge(
            received_df,
            left_on=left_on,
            right_on=right_on,
            how='inner'
        )
        
        expected_df_shape = expected_df.shape
        received_df_shape = received_df.shape
        merged_df_shape = merged_df.shape
        
        if expected_df_shape[0] != merged_df_shape[0]:
            self.LOGGER.error(f"Shape error. Expected: {expected_df_shape[0]:,} | Received: {received_df_shape[0]:,} | Merged: {merged_df_shape[0]:,}")
            raise Exception("Shape error.")
        
        self.LOGGER.debug(f"Shape has NO error.")
        
        for feature in self.features:
            
            self.LOGGER.debug(f'Validating feature {feature.name}...')
            
            expected_feature_name: str = f'expected.{feature.name}'
            received_feature_name: str = f'received.{feature.name}'
            
            # get err data
            error_data_cols \
                = [f'expected.{col}' for col in self.merge_on.list_features_name()]
                # + [expected_feature_name, received_feature_name]
                
            error_data = merged_df[error_data_cols]
            error_data = error_data.rename({
                    f'expected.{merge_feature_name}': merge_feature_name
                    for merge_feature_name in self.merge_on.list_features_name()
                },
                axis=1
                )
            error_data[expected_feature_name] = merged_df[expected_feature_name].copy()
            error_data[received_feature_name] = merged_df[received_feature_name].copy()
            
            if feature.type is FeatureType.DATETIME \
            or feature.type is FeatureType.DATETIMEUTC:
                error_data['error'] = error_data[received_feature_name] - error_data[expected_feature_name]
                error_data['tolerance'] = self.datetime_error_tolerance.get_tolerance_by_feature(
                    feature=feature
                )
                error_data['valid'] = error_data['error'].abs() <= error_data['tolerance']
                
            elif feature.type is FeatureType.INT8 \
            or feature.type is FeatureType.INT16 \
            or feature.type is FeatureType.INT32 \
            or feature.type is FeatureType.INT64 \
            or feature.type is FeatureType.UINT8 \
            or feature.type is FeatureType.UINT16 \
            or feature.type is FeatureType.UINT32 \
            or feature.type is FeatureType.UINT64 \
            or feature.type is FeatureType.FLOAT32 \
            or feature.type is FeatureType.FLOAT64:
                error_data['error'] = error_data[received_feature_name] - error_data[expected_feature_name]
                error_data['tolerance'] = self.numeric_error_tolerance.get_tolerance_by_feature(
                    feature=feature
                )
                error_data['valid'] = error_data['error'].abs() <= error_data['tolerance']
                
            else:
                error_data['error'] = error_data[received_feature_name] != error_data[expected_feature_name]
                error_data['tolerance'] = None
                error_data['valid'] = error_data['error'] == False
            
            if (error_data['valid'] == False).sum() > 0:
                
                if self.error_log_level == 1:
                    error_txt = error_data[~error_data["valid"]].to_string()
                    self.LOGGER.error(f'Validation Error:\n{error_txt}')
                    
                elif self.error_log_level == 2:
                    error_txt = error_data.to_string()
                    self.LOGGER.error(f'Validation Error:\n{error_txt}')
                
                else:
                    raise Exception(f"Provided error_log_level is invalid: {self.error_log_level}")
                    
                raise Exception("Validation Error.")
            
            self.LOGGER.debug(f'Validating feature {feature.name}... Done!')
        
    def validate(
        self,
        payload: Any,
        expected: Any,
    ):
        self.LOGGER.debug("Parsing expected data to dataframe...")
        expected_df = self.parse_expected_data_to_dataframe(
            expected=expected
        )
        expected_df = expected_df[self._full_features_name]
        self.LOGGER.debug("Parsing expected data to dataframe... Done!")
        
        self.LOGGER.debug("Parsing payload data to dataframe...")
        payload_df = self.parse_payload_data_to_dataframe(
            payload=payload
        )
        self.LOGGER.debug("Parsing payload data to dataframe... Done!")
        
        if self.constructor is not None:
            self.LOGGER.debug("Constructing features from payload...")
            received_df = self.constructor.construct(
                dataframe=payload_df,
                LOGGER=self.LOGGER
            )
            self.LOGGER.debug("Constructing features from payload... Done!")
        else:
            received_df = payload_df
        received_df = received_df[self._full_features_name]
        
        self.LOGGER.debug("Validating...")
        self._validate(
            expected_df=expected_df,
            received_df=received_df,
        )
        self.LOGGER.debug("Validating... Done!")
    
        self.LOGGER.info("Feature Constructor Successfully Validated!")
        
    def parse_expected_data_to_dataframe(
        self,
        expected: Any
    ) -> DataFrame:
        raise NotImplementedError
    
    def parse_payload_data_to_dataframe(
        self,
        payload: Any,
    ) -> DataFrame:
        raise NotImplementedError


class JSONFeaturesValidator(BaseFeaturesValidator):
        
    def parse_expected_data_to_dataframe(
        self,
        expected: Any
    ) -> DataFrame:
        expected_df = DataFrame(expected)
        expected_df = FeaturesTyper(self.features).type(expected_df, LOGGER=self.LOGGER)
        expected_df = FeaturesTyper(self.merge_on).type(expected_df, LOGGER=self.LOGGER)
        return expected_df
    
    def parse_payload_data_to_dataframe(
        self,
        payload: Any
    ) -> DataFrame:
        return DataFrame(payload)


# class DataFrameFeaturesValidator(BaseFeaturesValidator):
        
#     def parse_expected_data_to_dataframe(
#         self,
#         expected: Any
#     ) -> DataFrame:
#         expected_df = FeaturesTyper(self.features).type(expected, LOGGER=self.LOGGER)
#         expected_df = FeaturesTyper(self.merge_on).type(expected_df, LOGGER=self.LOGGER)
#         return expected_df
    
#     def parse_payload_data_to_dataframe(
#         self,
#         payload: Any
#     ) -> DataFrame:
#         return received