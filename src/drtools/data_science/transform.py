

from drtools.logging import Logger, FormatterOptions
from pandas import DataFrame
from drtools.data_science.features import (
    BaseFeatureConstructor
)


class BaseTransformer:
    
    def transform(self, *args, **kwargs):
        raise NotImplementedError
    
    
class BaseFeaturesConstructorTransformer(BaseTransformer):
    
    def __init__(
        self,
        constructor: BaseFeatureConstructor,
        LOGGER: Logger=Logger(
            name="BaseFeatureTransformer",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ):
        self.constructor = constructor
        self.set_logger(LOGGER)
        
    def set_logger(self, LOGGER):
        self.LOGGER = LOGGER
        
    def parse_payload_to_dataframe(
        self, 
        payload,
        **kwargs
    ) -> DataFrame:
        raise NotImplementedError
        
    def parse_consctructor_output_dataframe(
        self, 
        dataframe: DataFrame, 
        **kwargs
    ) -> DataFrame:
        raise NotImplementedError
    
    def transform(self, payload, LOGGER: Logger=None, **kwargs):
        
        if LOGGER is not None:
            self.set_logger(LOGGER)
            
        self.LOGGER.debug("Parsing payload to DataFrame...")
        parsed_payload: DataFrame = self.parse_payload_to_dataframe(
            payload,
            **kwargs,
        )
        self.LOGGER.debug("Parsing payload to DataFrame... Done!")
        
        self.LOGGER.debug("Constructing features...")
        constructed_features: DataFrame = self.constructor.construct(
            parsed_payload,
            LOGGER=self.LOGGER,
            **kwargs,
        )
        self.LOGGER.debug("Constructing features... Done!")
        
        self.LOGGER.debug("Parsing constructor output to DataFrame...")
        output: DataFrame = self.parse_consctructor_output_dataframe(
            constructed_features, 
            **kwargs
        )
        self.LOGGER.debug("Parsing constructor output to DataFrame... Done!")
        
        return output