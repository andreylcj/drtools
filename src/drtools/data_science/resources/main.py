""" 
This module was created to archive useful ways to handle 
with Model transforming.

"""


from pandas import DataFrame
import pandas as pd
import os
import numpy as np
from datetime import datetime
import logging
from drtools.utils import (
    ValueRestrictions, ValueRestrictionsAsJson
)
from drtools.decorators import start_end_log
from drtools.file_manager import (
    create_directories_of_path, rm_file, 
    list_path_of_all_files_inside_directory
)
import drtools.data_science.data_handle as DataHandle
import drtools.utils as Utils
from typing import List, Union, Optional, Dict, TypedDict, Any
from types import FunctionType
from drtools.logs import Log
from drtools.utils import list_ops
from pathlib import Path
from pydbml import PyDBML
from abc import ABC
import ast
from drtools.data_science.data_load import read_dir_as_df, read_as_df
import joblib


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


class CategoricalConditions(TypedDict):
    accepted_categories: Dict[str, int]
    accept_others: Optional[bool]
    accept_empty: Optional[bool]
    extra_categories: Optional[Dict[str, int]]


class SimpleFeature(TypedDict):
    name: str
    type: Union[VarChar, Str, Int, Float, Datetime, TimeStamp]    

    
class Feature(SimpleFeature):
    description: Union[Categorical, Numerical]
    conditions: Union[CategoricalConditions, ValueRestrictionsAsJson]
    observation: str


class TypeColumm(TypedDict):
    name: str
    typeraze: FunctionType
    

class Config(TypedDict):
    insert_oneFile: bool
    insert_fileExtension: bool
    typeraze_customTreatment: List[TypeColumm]


def filter_categorical(
    df: DataFrame,
    col: str,
    conditions: dict
) -> DataFrame:
    """Filter categorical data based on conditions.

    Parameters
    ----------
    df : DataFrame
        The DataFrame.
    col : str
        The column from where categories will be filtered.
    conditions : dict
        The conditions that will be applied when filtering categories.

    Returns
    -------
    DataFrame
        The filtered DataFrame.
    """
    
    if len(conditions) == 0:
        return df
    
    df[col] = np.where(pd.isnull(df[col]), df[col], df[col].astype(str))   
            
    accepted_categories = list(conditions.get('accepted_categories').keys())
    
    if conditions.get('accept_others', False):
        join_cat = list(df[col].dropna().unique())
        join_cat = Utils.list_ops(
            join_cat,
            accepted_categories
        )
        df = DataHandle.join_categories(
            df,
            col,
            join_cat,
            'OTHERS'
        )

    if conditions.get('accept_empty', False):
        df[col] = df[col].fillna('EMPTY')
        
    accepted_categories = accepted_categories + list(conditions.get('extra_categories', {}).keys())
    df = df[df[col].isin(accepted_categories)]
    
    return df


def filter_numerical(
    data: Union[DataFrame, np.array],
    cols: Union[List[int], List[str]],
    conditions: dict,
    as_numpy: bool=False
) -> Union[DataFrame, np.array]:
    """Filter numerical data based on conditions.

    Parameters
    ----------
    df : DataFrame
        The DataFrame.
    col : str
        The column from where numerical values will be filtered.
    conditions : dict
        The conditions that will be applied when filtering values.
    as_numpy : bool, Optional
        If True, will apply restrictions on a numpy matrix, 
        If False, will apply restrictions on a DataFrame, 
        by default False. 

    Returns
    -------
    DataFrame
        The filtered DataFrame.
    """
    if not as_numpy:
        if len(conditions) == 0:
            data[cols] = data.loc[:, cols].astype(float)
            return data
        value_restrictions = ValueRestrictions()
        value_restrictions.initialize_from_dict(conditions)
        data[cols] = data.loc[:, cols].astype(float)
        data = value_restrictions.restrict_df(data, cols)
    else:
        if len(conditions) == 0:
            # data[:, col] = data[:, col].astype(float)
            return data
        value_restrictions = ValueRestrictions()
        value_restrictions.initialize_from_dict(conditions)
        data = value_restrictions.restrict_numpy(data, cols)
        # data[:, col] = data[:, col].astype(float)
    return data


String = ['varchar', 'str']
Time = ['timestamp', 'datetime']
Int = ['int']
Float = ['float']
JSONB = ['JSONB', 'JSON']


def typeraze(
    dataframe: DataFrame,
    features: List[SimpleFeature],
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


class ModelCatalogueSingle(TypedDict):
    id: int
    created_at: datetime
    updated_at: datetime
    name: str
    version: str
    algorithm: str
    algorithm_infrastructure: Any
    description: str
    rules: str
    input_features: List[Feature]
    output_features: List[Feature]


class Model:
    """Class to handle model loading based on definition 
    on definition pattern presented on ModelCatalogue.
    
    Methods
    -------
    - extra_features_name()
    - output_features_name()
    - get_model_name()
    - cols_correct_order()
    - load_model()
    - save_model()
    - train()
    - predict()
    - one_hot_encoding()
    - label_encoding()
    - filter()
    - typeraze()
    """
    
    def __init__(
        self,
        model_catalogue_single: ModelCatalogueSingle,
  		LOGGER: Log=None,
        chained_assignment_log: bool=False
    ) -> None:
        """Init Model instance.

        Parameters
        ----------
        model_catalogue_single : ModelCatalogueSingle
            The model definitions.
        LOGGER : Log, optional
            The LOGGER instance to handle logs 
            , by default None
        chained_assignment_log : bool, optional
            If False, put pandas chained assignment equals None, 
            If True, do not change anything, by default False.
        """
        for k in model_catalogue_single:
            setattr(self, k, model_catalogue_single[k])
        self.LOGGER = logging if LOGGER is None else LOGGER
        if not chained_assignment_log:
            pd.options.mode.chained_assignment = None # default='warn'
            
    # @start_end_log('extra_features_name')
    def extra_features_name(self) -> List[str]:
        """Returns list of model extra columns names.

        Returns
        -------
        List[str]
            Model extra columns names.
        """
        return [feature['name'] for feature in self.extra_features]
    
    # @start_end_log('input_features_name')
    def input_features_name(self) -> List[str]:
        """Returns list of model input features name.

        Returns
        -------
        List[str]
            Model input features name.
        """
        return [feature['name'] for feature in self.input_features]
    
    # @start_end_log('output_features_name')
    def output_features_name(self) -> List[str]:
        """Returns list of model output features name.

        Returns
        -------
        List[str]
            Model output features name.
        """
        return [feature['name'] for feature in self.output_features]
    
    # @start_end_log('get_model_name')        
    def get_model_name(self) -> str:
        """Returns model name.

        Returns
        -------
        str
            Model name combining id, algorithm nickname, model name 
            and model version.
        """
        return f'{self.name}-{self.version}'
    
    # @start_end_log('cols_correct_order')        
    def cols_correct_order(
        self
    ) -> List[str]:
        """Returns list of all cols of model, including 
        extra columns, in correct order.

        Returns
        -------
        List[str]
            Model cols in correct order.
        """
        extra_features_name = self.extra_features_name()
        input_features_name = self.input_features_name()
        output_features_name = self.output_features_name()
        pretty_cols = list_ops(extra_features_name, input_features_name + output_features_name) \
            + input_features_name \
            + output_features_name
        return pretty_cols
    
    @start_end_log('load_model')
    def load_model(
        self,
        model_file_path: str,
        *args,
        **kwargs
    ) -> Any:
        """Load model from path and return model instance

        Parameters
        ----------
        model_file_path : str
            Path of model file.
        args : Tuple, optional
            All args inputs will be passed to load model 
            function, by default ().
        kwargs : Dict, optional
            All args inputs will be passed to load model 
            function, by default {}.

        Returns
        -------
        Any
            The model instance

        Raises
        ------
        Exception
            If model algorithm is invalid
        """
        
        self.LOGGER.info(f'Loading model {self.get_model_name()}...')        
        model = None        
        if self.model_algorithm == 'LightGBM':
            import lightgbm as lgb
            model = lgb.Booster(model_file=model_file_path, *args, **kwargs)
        elif self.model_algorithm == 'NeuralNetworks':
            from tensorflow import keras
            model = keras.models.load_model(model_file_path, *args, **kwargs)
        else:
            raise Exception(f'Algorithm {self.model_algorithm} is invalid.')        
        self.LOGGER.info(f'Loading model {self.get_model_name()}... Done!')        
        return model
    
    @start_end_log('save_model')
    def save_model(
        self,
        model_instance: Any,
        path: str,
        *args,
        **kwargs
    ) -> None:
        """Save model with path.

        Parameters
        ----------
        model_instance : Any
            Instance of desired model to save. 
        path : str
            The path to save model.
        args : Tuple, optional
            All args inputs will be passed to save model 
            function, by default ().
        kwargs : Dict, optional
            All args inputs will be passed to save model 
            function, by default {}.

        Returns
        -------
        None
            None is returned.

        Raises
        ------
        Exception
            If model algorithm is invalid
        """
        self.LOGGER.info(f'Saving model {self.get_model_name()}...')        
        # save_path = f'{project_root_path}/models/{self.get_model_name()}/model/{self.get_model_name()}'        
        if self.model_algorithm == 'LightGBM':
            create_directories_of_path(path)
            model_instance.save_model(filename=path, *args, **kwargs)
        elif self.model_algorithm == 'NeuralNetworks':
            create_directories_of_path(path)
            model_instance.save(path, *args, **kwargs)
        else:
            raise Exception(f'Algorithm {self.model_algorithm} is invalid.')        
        self.LOGGER.info(f'Saving model {self.get_model_name()}... Done!')
    
    @start_end_log('train')
    def train(
        self,
        model_instance: Any,
        *args,
        **kwargs
    ) -> Any:
        """Train model.

        Parameters
        ----------
        model_instance : Any
            Instance of desired model to train.
        args : Tuple, optional
            All args inputs will be passed to train 
            function, by default ().
        kwargs : Dict, optional
            All kwargs inputs will be passed to train 
            function, by default {}.

        Returns
        -------
        Any
            Returns different data for each algorithm. 

        Raises
        ------
        Exception
            If model algorithm is invalid
        """
        self.LOGGER.info(f'Training model {self.get_model_name()}...')                
        if self.model_algorithm == 'LightGBM':
            import lightgbm as lgb
            model_instance = lgb.train(*args, **kwargs)
            self.LOGGER.info(f'Training model {self.get_model_name()}... Done!')            
            return model_instance
        elif self.model_algorithm == 'NeuralNetworks':
            history = model_instance.fit(*args, **kwargs)
            self.LOGGER.info(f'Training model {self.get_model_name()}... Done!')            
            return model_instance, history
        else:
            raise Exception(f'Algorithm {self.model_algorithm} is invalid.')
    
    @start_end_log('predict')
    def predict(
        self,
        model_file_path: str,
        X: Any,
        *args,
        **kwargs
    ) -> Any:    
        """Predict data.

        Parameters
        ----------
        model_file_path : str
            Path of model file.
        X : Any
            X data to predict.
        args : Tuple, optional
            All args inputs will be passed to predict 
            function, by default ().
        kwargs : Dict, optional
            All kwargs inputs will be passed to predict 
            function, by default {}.

        Returns
        -------
        Any
            Returns different data for each algorithm. 

        Raises
        ------
        Exception
            If model algorithm is invalid
        """    
        self.LOGGER.info(f'Predicting data for model {self.get_model_name()}...')  
        model_instance = self.load_model(model_file_path)
        if self.model_algorithm == 'LightGBM':
            y_pred = model_instance.predict(X, *args, **kwargs)
        elif self.model_algorithm == 'NeuralNetworks':
            y_pred = model_instance.predict(X, *args, **kwargs)
            y_pred = y_pred.reshape(1, -1)[0]
        else:
            raise Exception(f'Algorithm {self.model_algorithm} is invalid.')        
        self.LOGGER.info(f'Predicting data for model {self.get_model_name()}... Done!')        
        return y_pred
    
    @start_end_log('one_hot_encoding')
    def one_hot_encoding(
        self,
        dataframe: DataFrame,
        encode_cols: List[str]
    ) -> DataFrame:
        """One hot encode variables, drop original column that 
        generate encoded and drop dummy cols that is not present 
        on the input features.
        
        Parameters
        ----------
        dataframe : DataFrame
            DataFrame containing data to encode.
        encode_cols : List[str]
            List with name of columns to one hot encode.
            
        Returns
        -------
        DataFrame
            The DataFrame containing encoded columns.
        """
        df = dataframe.copy()        
        for col in encode_cols:
            curr_features = [
                feature for feature in self.input_features
                if feature.get('observation', None) == col
            ]
            dummies = pd.get_dummies(df[col], prefix=col)
            drop_cols = list_ops(dummies.columns, self.input_features_name())
            df = pd.concat([df, dummies], axis=1)
            drop_self_col = list_ops([col], self.extra_features_name())            
            df = df.drop(drop_cols + drop_self_col, axis=1)            
            # insert feature that not has on received dataframe
            for curr_feature in curr_features:
                if curr_feature['name'] not in df.columns:
                    df[curr_feature['name']] = 0
        return df
    
    @start_end_log('label_encoding')
    def label_encoding(
        self,
        dataframe: DataFrame,
        astype_category: bool=False
    ) -> DataFrame:
        """Label encode variables.
        
        Parameters
        ----------
        dataframe : DataFrame
            DataFrame containing data to encode.
        astype_category : bool, optional
            If True, in set categorical columns to type "category", 
            If False, will encode values with integers, by default False
            
        Returns
        -------
        DataFrame
            The DataFrame containing encoded columns.
        """
        df = dataframe.copy()
        categorical = {
            feature['name']: feature['conditions']
            for feature in self.input_features 
            if feature.get('description', None) == Categorical
        }        
        encode = {
            col: {
                **conditions.get('accepted_categories'),
                **conditions.get('extra_categories', {})
            }
            for col, conditions in categorical.items()
        }        
        temp_encode = {
            alias: {
                **conditions.get('accepted_categories'),
                **conditions.get('extra_categories', {})
            }
            for conditions in categorical.values()
            for alias in conditions.get('aliases', [])
        }        
        encode = { **encode, **temp_encode }            
        for col in df.columns:
            if col in encode:
                if astype_category:
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace(encode[col])        
        return df
    
    @start_end_log('filter')
    def filter(
        self,
        dataframe: DataFrame,
        filter_only: List[str]=None,
        ignore_features: List[str]=[],
        ignore_categorical: bool=False,
        ignore_numerical: bool=False,
        custom: List[Feature]=None,
        verbosity: bool=True,
        as_numpy: bool=False,
    ) -> DataFrame:
        """Filter data to acceptable data to model.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing data to be filtered.
        filter_only : List[str], optional
            Filter only features that has name listed on this variable 
            , by default None.
        ignore_features : List[str], optional
            Ignore all features that has name listed on this variable 
            , by default [].
        ignore_categorical : bool, optional
            If True, ignore all Categorical features, by default False.
        ignore_numerical : bool, optional
            If True, ignore all Numerical features, by default False.
        custom : List[Feature], optional
            Custom list of features to filter. If provide, 
            will filter only the features listed on this variable, by default None.
        verbosity : bool, optional
            If True, verbose function statements, 
            if False, do not verbose, by default True.
        as_numpy : bool, optional
            If True, will treat numerical data as numpy arrays, 
            this option can decrease the computation time a lot 
            , by default False.

        Returns
        -------
        DataFrame
            The filtered DataFrame.

        Raises
        ------
        Exception
            If "description" fielf of Feature is invalid.
        """
        
        df = dataframe.copy()
        
        try:
            self.LOGGER.set_verbosity(verbosity)
        except:
            pass
        
        all_features = self.input_features + self.output_features
        
        filter_features = all_features
        if custom is not None:
            filter_features = custom
            
        categorical_features = []
        unique_numerical_conditions = []
        
        for feature in filter_features:
            
            if feature["name"] in ignore_features:
                self.LOGGER.debug(f'Ignore feature {feature["name"]}. Skipping...')
                continue
            
            if filter_only is not None:
                if feature["name"] not in filter_only:
                    self.LOGGER.debug(f'Feature {feature["name"]} not on Filter Only. Skipping...')
                    continue
            
            if feature["description"] == Numerical:
                
                if ignore_numerical:
                    self.LOGGER.debug(f'Ignore Numerical feature {feature["name"]}. Skipping...')
                    continue
                
                already_inserted = False
                for idx, row in enumerate(unique_numerical_conditions):
                    conditions = row[0]
                    if feature["conditions"] == conditions:
                        unique_numerical_conditions[idx][1].append(feature)
                        already_inserted = True
                        break
                    
                if not already_inserted:
                    unique_numerical_conditions.append(
                        [feature["conditions"], [feature]]
                    )
                    
                
            elif feature["description"] == Categorical:
                
                if ignore_categorical:
                    self.LOGGER.debug(f'Ignore Categorical feature {feature["name"]}. Skipping...')
                    continue
            
                categorical_features.append(feature)
                
            else:
                raise Exception(f'Feature {feature} has an invalid "description" field.')
            
        self.LOGGER.debug('Transforming and filtering Numerical variables...')
        
        numerical_unique_conditions_len = len(unique_numerical_conditions)        
        df_columns = df.columns
            
        self.LOGGER.debug('Generating numerical data...')
        if as_numpy:
            numerical_data = df.values
            numerical_data[pd.isnull(numerical_data)] = np.nan
        else:
            numerical_data = df
        self.LOGGER.debug('Generating numerical data... Done!')
            
        for idx, row in enumerate(unique_numerical_conditions):
            
            conditions = row[0]
            features = row[1]
            curr_features_name = [x['name'] for x in features]
            count = idx + 1
                
            self.LOGGER.debug('')
            self.LOGGER.debug(f'({count:,}/{numerical_unique_conditions_len:,}) Filtering {len(curr_features_name):,} numerical features...')
            self.LOGGER.debug(f'Features: {curr_features_name}...')
            self.LOGGER.debug(f'Conditions: {conditions}')
            
            from_shape = numerical_data.shape
                            
            if as_numpy:
                numerical_data = filter_numerical(
                    numerical_data, 
                    [df_columns.get_loc(name) for name in curr_features_name], 
                    conditions,
                    as_numpy=True
                )
            else:
                numerical_data = filter_numerical(
                    numerical_data, 
                    curr_features_name, 
                    conditions
                )
                
            to_shape = numerical_data.shape
            self.LOGGER.debug(f'Filter: from shape ({from_shape[0]:,}, {from_shape[1]:,}) to shape ({to_shape[0]:,}, {to_shape[1]:,})')
            self.LOGGER.debug(f'({count:,}/{numerical_unique_conditions_len:,}) Filtering {len(curr_features_name):,} numerical features... Done!')
            
            
        if as_numpy:
            self.LOGGER.debug('Generating DataFrame after filter numerical...')
            df = pd.DataFrame(
                numerical_data,
                columns=df_columns
            )
            self.LOGGER.debug('Generating DataFrame after filter numerical... Done!')
        else:
            df = numerical_data
            
        self.LOGGER.debug('Transforming and filtering Numerical variables... Done!')
        self.LOGGER.debug('')
        self.LOGGER.debug('Transforming and filtering Categorical variables...')
        
        categorical_features_len = len(categorical_features)
        
        for idx, feature in enumerate(categorical_features):
            
            count = idx + 1
                
            self.LOGGER.debug('')
            self.LOGGER.debug(f'({count:,}/{categorical_features_len:,}) Filtering {feature["name"]}...')
            self.LOGGER.debug(f'Conditions: {feature["conditions"]}')            
            
            from_shape = df.shape
            df = filter_categorical(df, feature["name"], feature["conditions"])                
            to_shape = df.shape
                
            self.LOGGER.debug(f'Col <{feature["name"]}> filter from shape ({from_shape[0]:,}, {from_shape[1]:,}) to shape ({to_shape[0]:,}, {to_shape[1]:,})')                    
            self.LOGGER.debug(f'({count:,}/{categorical_features_len:,}) Filtering {feature["name"]}... Done!')
            
        self.LOGGER.debug('Transforming and filtering Categorical variables... Done!')
            
        try:
            self.LOGGER.reset_verbosity()
        except:
            pass
            
        return df
    
    @start_end_log('typeraze')
    def typeraze(
        self,
        dataframe: DataFrame,
        as_model_input: bool=False,
        type_only: List[str]=None,
        ignore_features: List[str]=[],
        custom: List[Feature]=None,
        **kwargs
    ) -> DataFrame:
        """Type model variables.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing the data to be typed.
        as_model_input : bool, optional
            If True, when Feature has "type" equals "int", treat as "float", 
            if False, when Feature has "type" equals "int", treat as "int", 
            by default False.
        type_only : List[str], optional
            Type only features that has name listed on this variable 
            , by default None.
        ignore_features : List[str], optional
            Ignore all features that has name listed on this variable 
            , by default [].
        custom : List[Feature], optional
            Custom list of features to type. If provide, 
            will type only the features listed on this variable, by default None.
        kwargs : Dict, optional
            Arguments that will be passed to read_dir_as_df() 
            function and to typeraze() function 
            , by default {}.

        Returns
        -------
        DataFrame
            The typed DataFrame.
        """
        all_features_name = self.cols_correct_order()
        
        input_features_name = self.input_features_name()
        output_features_name = self.output_features_name()
        extra_features_name = self.extra_features_name()
        
        all_features = []
        for feature_name in all_features_name:
            if feature_name in input_features_name:
                for feature in self.input_features:
                    if feature_name == feature['name']:
                        all_features.append(feature)
                        break
            elif feature_name in output_features_name:
                for feature in self.output_features:
                    if feature_name == feature['name']:
                        all_features.append(feature)
                        break
            elif feature_name in extra_features_name:
                for feature in self.extra_features:
                    if feature_name == feature['name']:
                        all_features.append(feature)
                        break
            else:
                raise Exception(f'Feature "{feature_name} not present on input features, output features and extra columns"')
        
        if type_only is None:
            type_only = all_features_name
        
        features = [
            {
                'name': col['name'], 
                'type': col['type'] if not as_model_input \
                    else 'float' if col['type'] == 'int' \
                        else col['type'] 
            } 
            for col in all_features
            if col['name'] not in ignore_features \
            and col['name'] in type_only
        ]
        
        if custom is not None:
            features = [
                {'name': feature['name'], 'type': feature['type']}
                for feature in custom
            ]
        df = dataframe.copy()
        df = typeraze(dataframe=df, features=features, LOGGER=self.LOGGER, **kwargs)
        return df


class Database(ABC):
    """Abstract class to handle database loading based on default 
    infrastructure.
    
    Methods
    -------
    - get_col_names()
    - typeraze()
    - load_db_as_df()
    - _save_data()
    - insert()
    - delete_by_id()
    - update_by_id()
    """
    
    CONFIG: Config = Config({
        'insert_oneFile': False,
        'insert_fileExtension': 'parquet',
        'typeraze_customTreatment': []
    })
    
    def __init__(
        self,
        database_dbml_path: str,
        database_root_path: str,
  		LOGGER: Log=None,
    ) -> None:
        """Init Database instance.

        Parameters
        ----------
        database_dbml_path : str
            Path to .dbml database documentation.
        database_root_path : str
            Path to root of database.
        LOGGER : Log, optional
            The LOGGER instance to handle logs 
            , by default None
        """
        assert hasattr(self, 'SCHEMA')
        assert hasattr(self, 'TABLE')  
        self.database_dbml_path = database_dbml_path
        self.database_root_path = database_root_path
        parsed = PyDBML(Path(database_dbml_path))
        try:
            self.table = parsed[f'{self.SCHEMA}.{self.TABLE}']
            self.col_names = [col.name for col in self.table.columns]
        except:
            self.enum = [
                enum 
                for enum in parsed.enums 
                if enum.name == self.TABLE \
                and enum.schema == self.SCHEMA
            ][0]
            self.col_names = []
            for col in self.enum.items:
                self.col_names.append(col.name)
                setattr(self, col.name, col)
            
            
        self.LOGGER = logging if LOGGER is None else LOGGER
        
    def get_col_names(self) -> List[str]:
        """Get database column names.

        Returns
        -------
        List[str]
            List containing database column names.
        """
        return self.col_names
    
    def get_manual_col_names(self) -> List[str]:
        """Get database manual column names.

        Returns
        -------
        List[str]
            List containing database column names.
        """
        resp = list_ops(
            self.col_names,
            ['id', 'created_at', 'updated_at']
        )
        return resp      
    
    def db_name(self) -> str:
        """Get name of database

        Returns
        -------
        str
            Name of database
        """
        return f'{self.SCHEMA}.{self.TABLE}'
    
    @start_end_log('typeraze')
    def typeraze(
        self,
        dataframe: DataFrame,
        dtypes: List[str]=None,
        ignore_dtypes: List[str]=[],
        verbosity: bool=False,
        **kwargs
    ) -> DataFrame:
        """Type database columns.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame contining the data to be typed.
        dtypes : List[str], optional
            Specify Dtypes to be handled, by default None
        ignore_dtypes : List[str], optional
            Ignore Dtypes when typing data, by default []
        verbosity : bool, optional
            If True, verbose functions statements, 
            if False, do not verbose, by default True

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
        useful_cols = list_ops(
            df.columns,
            self.col_names,
            ops='intersection'
        )
        df = df[useful_cols]
        
        features = [
            {'name': col.name, 'type': col.type} 
            for col in self.table.columns 
            if col.name in useful_cols
        ]
        
        custom_treatment = self.CONFIG.get('typeraze_customTreatment', [])
        
        try:
            self.LOGGER.set_verbosity(verbosity)
        except:
            pass
        
        df = typeraze(
            dataframe=df,
            features=features,
            dtypes=dtypes,
            ignore_dtypes=ignore_dtypes,
            custom_treatment=custom_treatment,
            LOGGER=self.LOGGER,
            **kwargs
        )
        
        try:
            self.LOGGER.reset_verbosity()
        except:
            pass
        
        return df
        
    # @start_end_log('load_db_as_df')
    def load_db_as_df(
        self, 
        typeraze: bool=False,
        verbosity: bool=False,
        **kwargs
    ) -> DataFrame:
        """Load local database data as DataFrame

        Parameters
        ----------
        typeraze : bool, optional
            If True, the data will be typed after 
            loaded. If False, the data will not be 
            typed after load, by default False
        kwargs : Dict, optional
            Arguments that will be passed to read_dir_as_df() 
            function and to typeraze() function 
            , by default {}.

        Returns
        -------
        DataFrame
            The local database data as DataFrame
        """
        self.LOGGER.info(f'Loading {self.db_name()}...')
        
        df = read_dir_as_df(
            Utils.join_path(
                self.database_root_path, self.SCHEMA, self.TABLE
            ),
            **kwargs
        )
        
        if df is None:
            real_cols = kwargs.get('usecols', None)
            if real_cols is None:
                real_cols = self.get_col_names()
            df = pd.DataFrame([], columns=real_cols)
          
        if typeraze:              
            df = self.typeraze(
                df,
                verbosity=verbosity,
                **kwargs
            )
        
        cols_correct_order = list_ops(self.get_col_names(), df.columns, ops='intersection')            
        df = df.loc[:, cols_correct_order]
            
        self.LOGGER.info(f'Loading {self.db_name()}... Done!')
            
        return df
    
    @start_end_log('_save_data')
    def _save_data(
        self,
        dataframe: DataFrame,
        filename: str,
        typeraze: bool=True,
        overwrite_if_exists: bool=False
    ) -> None:
        """Save data on local Database infrastructure handling types 
        of columns and filtering columns to fit with database Schema.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame to be saved.
        filename : str
            The desired filename to save the data.
        typeraze : bool, optional
            If the data needs to be typed before 
            save, by default True
        overwrite_if_exists : bool, optional
            If True, overwrite file with same name 
            if exists. If False, return Exception 
            if exist some file on local Database with 
            same filename, by default False

        Raises
        ------
        Exception
            If save path already exists.
        Exception
            If extension of file is not allowed.
        """
        
        df = dataframe.copy()
        df = df.loc[:, self.get_col_names()]
        if typeraze:
            df = self.typeraze(df, verbosity=False)
        df = df.loc[:, self.get_col_names()]
        
        save_path = Utils.join_path(
            self.database_root_path, self.SCHEMA, self.TABLE, filename
        )
        if not overwrite_if_exists:
            if os.path.exists(save_path):
                raise Exception(f'Path {save_path} already exists.')
        
        create_directories_of_path(save_path)
        
        if '.csv' in filename:
            df.to_csv(save_path, index=False)
            
        elif '.parquet' in filename:
            
            for col in df.columns:                
                real_col = [col1 for col1 in self.table.columns if col == col1.name][0]
                if real_col.type in JSONB:
                    df[col] = df[col].astype(str)
                    
            df.to_parquet(save_path, index=False)
            
        elif '.json' in filename:
            df.to_json(
                save_path,                
                orient='records', 
                date_format='iso',
                indent=4
            )
        else:
            raise Exception(f'Extension on {filename} not allow.')
    
    # @start_end_log('insert')
    def insert(
        self,
        dataframe: DataFrame
    ) -> List[int]:
        """Insert data on db. Handle data before insert. Typeraze and 
        get only columns that must be exist on db.

        Parameters
        ----------
        dataframe : DataFrame
            The data in DataFrame format to be inserted

        Returns
        -------
        List[int]
            The inserted ids.
        """
        
        self.LOGGER.info(f'Inserting data on {self.db_name()}...')
        
        df = dataframe.copy()
        
        if df.shape[0] == 0:
            return []
        
        already_on_db = self.load_db_as_df(
            usecols=['id']
        )
        
        if already_on_db.shape[0] == 0:
            max_id = 0
        else:
            max_id = already_on_db.id.max()
        
        df['id'] = max_id + np.arange(len(df)) + 1
        inserted_ids = df['id'].values
        df['created_at'] = datetime.now().isoformat()
        df['updated_at'] = datetime.now().isoformat()
        
        # insert data
        default_extension = 'parquet'
        extension = self.CONFIG.get('insert_fileExtension', None)
        extension = default_extension if extension is None else extension
        filename = f'{self.SCHEMA}_{self.TABLE}-{datetime.now().isoformat()}.{extension}'
        
        if self.CONFIG.get('insert_oneFile', False):
            
            already_on_db = self.load_db_as_df()
            df = pd.concat(
                [already_on_db, df], 
                axis=0,
                ignore_index=True
            )
            self._save_data(df, filename)
            
            data_path = Utils.join_path(
                self.database_root_path, self.SCHEMA, self.TABLE
            )
            paths = list_path_of_all_files_inside_directory(data_path)
            for path in paths:
                if path.endswith(filename):
                    continue
                rm_file(
                    path,
                    ignore_if_path_not_exists=True
                )
        else:
            self._save_data(df, filename)
        
        self.LOGGER.info(f'Inserting data on {self.db_name()}... Done!')
        
        return sorted(inserted_ids)
    
    def insert_if_not_exists(
        self,
        dataframe: DataFrame
    ) -> List[int]:
        """Insert data on db only if not exists. Handle data before insert. Typeraze and 
        get only columns that must be exist on db.

        Parameters
        ----------
        dataframe : DataFrame
            The data in DataFrame format to be inserted

        Returns
        -------
        List[int]
            The inserted ids.
        """
        
        
        df = dataframe.copy()
        
        if df.shape[0] == 0:
            return []        
        
        self.LOGGER.info(f'Filtering already inserted...')
        original_shape = df.shape
        necessary_cols = list_ops(
            self.get_col_names(),
            ['id', 'created_at', 'updated_at']
        )
        df = df.loc[:, necessary_cols]
        df['md5_of_row'] = df.apply(lambda x: joblib.hash(x.values), axis=1)
        intersection_data = self.load_db_as_df(
            chunksize=400 * 10**3,
            process_chunk=lambda chunk_df: chunk_df[
                chunk_df.apply(
                    lambda x: joblib.hash(x.values), axis=1
                ).isin(df.md5_of_row.values)
            ]
        )
        intersection_data['md5_of_row'] = intersection_data.apply(
            lambda x: joblib.hash(x.values), axis=1
        )
        df = df[~df.md5_of_row.isin(intersection_data.md5_of_row.values)]
        df = df.drop('md5_of_row', axis=1)
        del intersection_data
        final_shape = df.shape
        self.LOGGER.info(f'{(original_shape[0] - final_shape[0]):,} rows was already inserted.')
        self.LOGGER.info(f'{final_shape[0]:,} rows to insert.')
        self.LOGGER.info(f'Filtering already inserted... Done!')
        
        # return df
        
        inserted_ids = self.insert(
            dataframe=df
        )
        
        return inserted_ids
        
    @start_end_log('delete_by_id')
    def delete_by_id(
        self,
        ids: List[int]
    ) -> List[int]:
        """Delete data from local Database by ids.

        Parameters
        ----------
        ids : List[int]
            The list of ids to be deleted.

        Returns
        -------
        List[int]
            The list of ids that was deleted.
        """
        
        data_path = Utils.join_path(
            self.database_root_path, self.SCHEMA, self.TABLE
        )
        paths = list_path_of_all_files_inside_directory(data_path)
        for path in paths:
            df = read_as_df(path)
            old_shape = df.shape
            df = df[~df.id.isin(ids)]
            new_shape = df.shape
            filename = os.path.basename(path)
            
            if old_shape[0] == new_shape[0]:
                continue
            
            self._save_data(
                df, 
                filename, 
                typeraze=False,
                overwrite_if_exists=True
            )
        
        return sorted(list(ids))
    
    @start_end_log('update_by_id')
    def update_by_id(
        self,
        dataframe: DataFrame
    ) -> List[int]:
        """Update data from local Database with id and data 
        from received DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            The data in DataFrame format to be inserted

        Returns
        -------
        List[int]
            The updated ids.
        """
        
        df = dataframe.copy()
        
        if df.shape[0] == 0:
            return []
        
        intersect_cols = list_ops(df.columns, self.get_col_names(), ops='intersection')
        intersect_cols = list_ops(intersect_cols, ['created_at', 'updated_at'])
        df = df.loc[:, intersect_cols]
        intersect_cols = intersect_cols + ['updated_at']
        df['updated_at'] = datetime.now().isoformat()
        
        if 'id' not in intersect_cols:
            raise Exception('Column id must be on DataFrame.')
        
        if df.id.values.shape[0] != df.id.unique().shape[0]:
            seen = set()
            dupes = [x for x in df.id.values if x in seen or seen.add(x)]
            dupes_df = df[df.id.isin(dupes)]
            self.LOGGER.info(dupes_df)
            raise Exception(f'There are duplicated ids on received data. Duplicate ids: {dupes}')
        
        data_path = Utils.join_path(
            self.database_root_path, self.SCHEMA, self.TABLE
        )
        paths = list_path_of_all_files_inside_directory(data_path)
        
        updated_ids = []
        
        for path in paths:
            database = read_as_df(path)
            
            ids_intersection = list_ops(
                database.id.values,
                df.id.values, 
                ops='intersection'
            )
            
            complementary_df = database.loc[~database.id.isin(ids_intersection)]
            database = database.loc[database.id.isin(ids_intersection)]
            
            curr_updated_ids = database.id.values.copy()
            
            drop_cols = list_ops(self.get_col_names(), df.columns, ops='intersection')
            drop_cols = list_ops(drop_cols, ['id'])
            
            database = database.drop(drop_cols, axis=1)
            database = database.merge(df, on=['id'], how='inner')

            database = database.loc[:, self.get_col_names()]
            database = pd.concat([
                complementary_df[self.get_col_names()], 
                database[self.get_col_names()]
            ], axis=0)
            database = database.sort_values('id').reset_index(drop=True)
 
            updated_ids = updated_ids + list(curr_updated_ids)
            
            filename = os.path.basename(path)
            
            self._save_data(
                database, 
                filename, 
                typeraze=True,
                overwrite_if_exists=True
            )
            
        not_updated_ids = list_ops(
            df.id.values,
            updated_ids
        )
        
        if len(not_updated_ids) > 0:
            not_updated_ids = sorted(not_updated_ids)
            self.LOGGER.debug(f'Not updated ids: {not_updated_ids}')
            
                    
        return sorted(updated_ids)
    
    def get_by_id(
        self,
        id: int,
        as_dataframe: bool=True
    ):
        df = self.load_db_as_df(
            chunksize=400 * 10**3,
            process_chunk=lambda chunk: chunk[chunk.id == id]
        )
        if as_dataframe:
            return df
        else:
            single = df.to_dict(orient='records')
            return single[0]
    

class ModelCatalogueWhen(TypedDict):
    id: int
    input_features: List[str]
    output_features: List[str]


First = 'first'
All = 'all'

    
class ModelCatalogue(Database):
    
    SCHEMA = 'model'
    
    TABLE = 'catalogue'
    
    @start_end_log('get_model')
    def get_model(
        self,
        when: ModelCatalogueWhen,
        returns: Union[First, All]=First
    ) -> Union[ModelCatalogueSingle, List[ModelCatalogueSingle]]:
        """Get model as dict from model.catalogue database by id.

        Parameters
        ----------
        when : ModelCatalogueWhen
            Conditions that model must satisfy.
        returns : Union[First, All], optional
            If "first", will returns only the first match of criterias, 
            If "all", will returns all models that match the criterias, 
            by default First

        Returns
        -------
        Union[Dict, List[Dict]]
            Dict representing the model or List of model dicts.

        Raises
        ------
        Exception
            If returns option is invalid.
        """
        df = self.load_db_as_df(typeraze=True)
        if 'id' in when:
            model_definition = df[df.id == when['id']]
        model_results = model_definition.to_dict(orient='records')
        if returns == First:
            return model_results[0]
        elif returns == All:
            return model_results
        else:
            raise Exception(f'Given value {returns} for "returns" variable is invalid.')
