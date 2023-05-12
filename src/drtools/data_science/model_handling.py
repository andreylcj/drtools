

from pandas import DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from drtools.decorators import start_end_log
from drtools.file_manager import (
    create_directories_of_path
)
from typing import List, TypedDict, Any, Union
from drtools.logs import Log
from drtools.utils import (
    ValueRestrictions, list_ops
)
from drtools.data_science.data_handle import join_categories
from drtools.data_science.general import typeraze
from drtools.data_science.features_handle import (
    ExtendedFeatureJSON, Categorical, Numerical
)


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
        join_cat = list_ops(
            join_cat,
            accepted_categories
        )
        df = join_categories(
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
    input_features: List[ExtendedFeatureJSON]
    output_features: List[ExtendedFeatureJSON]


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