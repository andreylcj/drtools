

from typing import Any, Callable
from drtools.logging import Logger, FormatterOptions
from drtools.airflow.types import Date
from datetime import datetime, timedelta


def get_relative_date_method(diff: int) -> Date:
    date = datetime.now()+ timedelta(diff)
    date: Date = date.strftime("%Y-%m-%d")
    return date

    
class Workflow:
    
    NAME: str = None
    
    def __init__(
        self,
        get_relative_date: Callable=get_relative_date_method,
        LOGGER: Logger=Logger(
            name="Workflow",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_thread_name=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        self.get_relative_date = get_relative_date
        self.LOGGER = LOGGER
    
    @property
    def name(self) -> str:
        return self.NAME or self.__class__.__name__

    
class ToThroughFromETLWorkflow(Workflow):
    
    To: To = None
    Through: Through = None
    From: From = None
    
    def __init__(self, *args, **kwargs) -> None:
        assert self.To is not None, \
            "Static attribute To must be set."
        assert self.Through is not None, \
            "Static attribute Through must be set."
        assert self.From is not None, \
            "Static attribute From must be set."
        super(ToThroughFromETLWorkflow, self).__init__(*args, **kwargs)
    
    def extract(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def transform(self, extract_response: Any, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def load(self, transform_response: Any, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def run(self, *args, **kwargs):
        self.LOGGER.info("Extracting...")
        self.extract_response = self.extract(*args, **kwargs)
        self.LOGGER.info("Extracting... Done!")
        
        self.LOGGER.info("Transforming...")
        self.transform_response = self.transform(self.extract_response, *args, **kwargs)
        self.LOGGER.info("Transforming... Done!")
        
        self.LOGGER.info("Loading...")
        self.load_response = self.load(self.transform_response, *args, **kwargs)
        self.LOGGER.info("Loading... Done!")
    
    
class CustomWorkflow(Workflow):
    
    def construct(self, *args, **kwargs) -> Callable:
        raise NotImplementedError