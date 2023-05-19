

from pandas import DataFrame
import pandas as pd
from datetime import datetime
import logging
from drtools.decorators import start_end_log
from drtools.file_manager import (
    create_directories_of_path
)
from typing import List, TypedDict, Any, Dict
from drtools.logs import Log, FormatterOptions
from drtools.utils import (
    list_ops
)
from drtools.data_science.features_handle import (
    ExtendedFeatureJSON, Categorical, Features, Feature
)
from enum import Enum


# class ModelCatalogueSingle(TypedDict):
#     id: int
#     created_at: datetime
#     updated_at: datetime
#     name: str
#     version: str
#     algorithm: str
#     algorithm_infrastructure: Any
#     description: str
#     rules: str
#     input_features: List[ExtendedFeatureJSON]
#     output_features: List[ExtendedFeatureJSON]


class Algorithm(Enum):
    LIGHTGBM = "LightGBM",
    NN = "Neural Network",
    
    @property
    def pname(self):
        return self.value[0]


class BaseModel:
    """Class to handle model loading based on definition 
    on definition pattern presented on ModelCatalogue.
    
    Methods
    -------
    - list_input_features_name()
    - list_extra_features_name()
    - list_output_features_name()
    - list_columns_correct_order()
    - load_model()
    - save_model()
    - train()
    - predict()
    """
    
    ALGORITHM: Algorithm = None
    
    @classmethod
    def init_from_json(
        cls, 
        LOGGER: Log=None, 
        chained_assignment_log: bool=False,
        **kwargs, 
    ):
        base_model = cls(
            name=kwargs.get('name', None),
            version=kwargs.get('version', None),
            algorithm_infrastructure=kwargs.get('algorithm_infrastructure', None),
            description=kwargs.get('description', None),
            rules=kwargs.get('rules', None),
            input_features=Features([
                Feature(**feature)
                for feature in kwargs.get('input_features', [])
            ]),
            output_features=Features([
                Feature(**feature)
                for feature in kwargs.get('output_features', [])
            ]),
            extra_features=Features([
                Feature(**feature)
                for feature in kwargs.get('extra_features', [])
            ]),
            training_information=kwargs.get('training_information', None),
            metrics=kwargs.get('metrics', None),
            LOGGER=LOGGER,
            chained_assignment_log=chained_assignment_log
        )
        return base_model
    
    def __init__(
        self,
        name: str,
        version: str,
        algorithm_infrastructure: Dict,
        description: str,
        rules: Dict,
        input_features: Features,
        output_features: Features,
        extra_features: Features,
        training_information: Dict={},
        metrics: List=[],
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
        self.name = name
        self.version = version
        self.algorithm_infrastructure = algorithm_infrastructure
        self.description = description
        self.rules = rules
        self.input_features = input_features
        self.output_features = output_features
        self.extra_features = extra_features
        self.training_information = training_information
        self.metrics = metrics
        self.LOGGER = Log(
                name=f"Model-{self.model_name}",
                formatter_options=FormatterOptions(
                    IncludeDate=True,
                    IncludeLoggerName=True,
                    IncludeLevelName=True,
                ),
            ) if LOGGER is None else LOGGER
        if not chained_assignment_log:
            pd.options.mode.chained_assignment = None # default='warn'
            
    def list_extra_features_name(self) -> List[str]:
        """Returns list of model extra columns names.

        Returns
        -------
        List[str]
            Model extra columns names.
        """
        return self.extra_features.list_features_name()
    
    # @start_end_log('input_features_name')
    def list_input_features_name(self) -> List[str]:
        """Returns list of model input features name.

        Returns
        -------
        List[str]
            Model input features name.
        """
        return self.input_features.list_features_name()
    
    # @start_end_log('output_features_name')
    def list_output_features_name(self) -> List[str]:
        """Returns list of model output features name.

        Returns
        -------
        List[str]
            Model output features name.
        """
        return self.output_features.list_features_name()
    
    @property
    def model_name(self) -> str:
        """Returns model name.

        Returns
        -------
        str
            Model name combining id, algorithm nickname, model name 
            and model version.
        """
        return f'{self.name}-{self.version}'
    
    @property
    def info(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version,
            'algorithm_infrastructure': self.algorithm_infrastructure,
            'description': self.description,
            'rules': self.rules,
            'input_features': self.input_features.info,
            'output_features': self.output_features.info,
            'extra_features': self.extra_features.info,
            'training_information': self.training_information,
            'metrics': self.metrics,
        }
    
    # @start_end_log('cols_correct_order')        
    def list_columns_correct_order(
        self
    ) -> List[str]:
        """Returns list of all cols of model, including 
        extra columns, in correct order.

        Returns
        -------
        List[str]
            Model cols in correct order.
        """
        extra_features_name = self.list_extra_features_name()
        input_features_name = self.list_input_features_name()
        output_features_name = self.list_output_features_name()
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
        pass
    
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
        pass
    
    @start_end_log('train')
    def train(
        self,
        model_instance: Any,
        *args,
        **kwargs
    ) -> Any:
        pass
    
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
        pass
    
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
    
    
class LightGBM(BaseModel):
    
    ALGORITHM: Algorithm = Algorithm.LIGHTGBM
    
    def load_model(self, model_file_path: str, *args, **kwargs) -> Any:
        self.LOGGER.info(f'Loading model {self.model_name}...')  
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_file_path, *args, **kwargs)
        self.LOGGER.info(f'Loading model {self.model_name}... Done!')  
        return model
    
    def save_model(self, model_instance: Any, path: str, *args, **kwargs) -> None:
        self.LOGGER.info(f'Saving model {self.model_name}...')
        create_directories_of_path(path)
        model_instance.save_model(filename=path, *args, **kwargs)
        self.LOGGER.info(f'Saving model {self.model_name}... Done!')
            
    def train(self, model_instance: Any, *args, **kwargs) -> Any:
        self.LOGGER.info(f'Training model {self.model_name}...')
        import lightgbm as lgb
        model_instance = lgb.train(*args, **kwargs)
        self.LOGGER.info(f'Training model {self.model_name}... Done!')            
        return model_instance
    
    def predict(self, model_file_path: str, X: Any, *args, **kwargs) -> Any: 
        self.LOGGER.info(f'Predicting data for model {self.model_name}...')  
        model_instance = self.load_model(model_file_path)
        y_pred = model_instance.predict(X, *args, **kwargs)
        self.LOGGER.info(f'Predicting data for model {self.model_name}... Done!')        
        return y_pred