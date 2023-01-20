""" 
This module was created to create transformers
which will receive some DataFrame and return a transformed
DataFrame.

"""


from abc import ABC, abstractmethod
from pandas.core.frame import DataFrame
from drtools.logs import Log
import logging


class BasicTransformer(ABC):
    """Abstract class to handle the Datasets transformations
        
    """
    
    def __init__(
        self,
        LOGGER: Log=None
    ) -> None:
        self.LOGGER = logging if LOGGER is None else LOGGER
    
    @abstractmethod
    def transform(
        self, 
        dataframe: DataFrame
    ) -> DataFrame:
        pass