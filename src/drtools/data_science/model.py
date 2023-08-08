

from drtools.decorators import start_end_log
from drtools.file_manager import (
    create_directories_of_path
)
from typing import List, Any, Dict, TypedDict, Tuple
from drtools.logging import Logger, FormatterOptions
from drtools.utils import (
    list_ops
)
from drtools.data_science.features_handling import (
    FeatureType,
    Feature, 
    Features, 
)
from enum import Enum
        
        
class MetricType(Enum):
    RMSE = "rmse", "Root Mean Square Error"
    
    @property
    def code(self):
        return self.value[0]


class AlgorithmType(Enum):
    LIGHTGBM = "LightGBM",
    NN = "Neural Network",
    
    @property
    def pname(self):
        return self.value[0]


class DatasetInfo:
    def __init__(
        self,
        shape: Tuple[int, int]
    ) -> None:
        self.shape = shape
    
    @property
    def info(self) -> Dict:
        return {
            'shape': self.shape,
        }

    
class Datasets:
    def __init__(
        self,
        train: DatasetInfo,
        validation: DatasetInfo,
        holdout: DatasetInfo,
    ) -> None:
        self.train = train
        self.validation = validation
        self.holdout = holdout
    
    @property
    def info(self) -> Dict:
        return {
            'train': self.train.info,
            'validation': self.validation.info,
            'holdout': self.holdout.info,
        }
        
        
class Metric:
    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        value
    ) -> None:
        self.name = name
        self.metric_type = metric_type
        self.value = value
    
    @property
    def info(self) -> Dict:
        return {
            'name': self.name,
            'metric': self.metric.code,
            'value': self.value,
        }
        
        
class Metrics:
    def __init__(
        self,
        metrics: List[Metric]
    ):
        self.metrics = metrics
    
    @property
    def info(self) -> List[Dict]:
        return [metric.info for metric in self.metrics]
    

class TrainingInformation:
    def __init__(
        self,
        datasets: Datasets
    ) -> None:
        self.datasets = datasets
    
    @property
    def info(self) -> Dict:
        return {
            'datasets': self.datasets.info,
        }
    

class ModelDefitinion:
    def __init__(
        self,
        algorithm: AlgorithmType,
        name: str,
        version: str,
        algorithm_infrastructure: Dict,
        description: str,
        rules: Dict,
        input_features: List[Feature],
        output_features: List[Feature],
        extra_features: List[Feature],
        training_information: TrainingInformation,
        metrics: Metrics
    ) -> None:
        self.algorithm = algorithm
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
         
    @property
    def pretty_cols(
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
        pretty_cols = list_ops(
            extra_features_name, 
            input_features_name + output_features_name
            ) \
            + input_features_name \
            + output_features_name
        return pretty_cols
    
    @property
    def name(self) -> str:
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
            'algorithm': self.algorithm.pname,
            'name': self.name,
            'version': self.version,
            'algorithm_infrastructure': self.algorithm_infrastructure,
            'description': self.description,
            'rules': self.rules,
            'input_features': self.input_features.info,
            'output_features': self.output_features.info,
            'extra_features': self.extra_features.info,
            'training_information': self.training_information.info,
            'metrics': self.metrics.info,
        }
    
    @classmethod
    def load_from_json(
        cls,
        **kwargs
    ):
        return cls(
            algorithm=kwargs['algorithm'],
            name=kwargs['name'],
            version=kwargs['version'],
            algorithm_infrastructure=kwargs.get('algorithm_infrastructure', None),
            description=kwargs.get('description', None),
            rules=kwargs.get('rules', None),
            input_features=Features([
                Feature(
                    name=feature['name'],
                    type=FeatureType.smart_instantiation(feature['type']),
                    **{k: v for k, v in feature.items() if k not in ['name', 'type']}
                )
                for feature in kwargs['input_features']
            ]),
            output_features=Features([
                Feature(
                    name=feature['name'],
                    type=FeatureType.smart_instantiation(feature['type']),
                    **{k: v for k, v in feature.items() if k not in ['name', 'type']}
                )             
                for feature in kwargs['output_features']
            ]),
            extra_features=Features([
                Feature(
                    name=feature['name'],
                    type=FeatureType.smart_instantiation(feature['type']),
                    **{k: v for k, v in feature.items() if k not in ['name', 'type']}
                )
                for feature in kwargs['extra_features']
            ]),
            training_information=kwargs.get('training_information', None),
            metrics=kwargs.get('metrics', None),
        )


class BaseModel:
    """Class to handle model loading based on definition 
    on definition pattern presented on ModelCatalogue.
    """
    
    ALGORITHM: AlgorithmType = None
    
    def __init__(
        self,
        model_definition: ModelDefitinion,
  		LOGGER: Logger=Logger(
            name="BaseModel",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        ),
    ) -> None:        
        self.model_definition = model_definition
        self.LOGGER = LOGGER
    
    @property
    def model_name(self) -> str:
        return self.model_definition.name
    
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
        model: Any,
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
        *args,
        **kwargs
    ) -> Any:
        pass
    
    @start_end_log('predict')
    def predict(
        self,
        model: Any,
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
    
    
class LightGbmModel(BaseModel):
    
    ALGORITHM: AlgorithmType = AlgorithmType.LIGHTGBM
    
    def load_model(self, model_file_path: str, *args, **kwargs) -> Any:
        self.LOGGER.debug(f'Loading model {self.model_name}...')  
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_file_path, *args, **kwargs)
        self.LOGGER.debug(f'Loading model {self.model_name}... Done!')  
        return model
    
    def save_model(self, model: Any, path: str, *args, **kwargs) -> None:
        self.LOGGER.debug(f'Saving model {self.model_name}...')
        create_directories_of_path(path)
        model.save_model(filename=path, *args, **kwargs)
        self.LOGGER.debug(f'Saving model {self.model_name}... Done!')
            
    def train(self, *args, **kwargs) -> Any:
        self.LOGGER.debug(f'Training model {self.model_name}...')
        import lightgbm as lgb
        model_instance = lgb.train(*args, **kwargs)
        self.LOGGER.debug(f'Training model {self.model_name}... Done!')            
        return model_instance
    
    def predict(self, model: str, X: Any, *args, **kwargs) -> Any: 
        self.LOGGER.debug(f'Predicting data for model {self.model_name}...')  
        # model_instance = self.load_model(model_file_path)
        y_pred = model.predict(X, *args, **kwargs)
        self.LOGGER.debug(f'Predicting data for model {self.model_name}... Done!')        
        return y_pred