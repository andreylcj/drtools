

from datetime import datetime
import traceback
from .common import BaseAttributesHandler
from typing import List, Any
from .assets import DataAsset, TransformerAsset, LoadAsset
from .resource import Resource


class Job(BaseAttributesHandler):
    """Base class for pipeline jobs that orchestrate assets and resources.

    Subclasses must implement execute_on_run() to define the job logic.
    run() wraps it with timing, logging, and error propagation.
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'DATA_ASSETS',
        'TRANSFORMER_ASSETS',
        'LOAD_ASSETS',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute_on_run(self):
        """Define the job execution logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def linear_materialize_all_data_assets(self) -> None:
        """Materialize all registered DATA_ASSETS sequentially in declaration order."""
        self.LOGGER.debug("Linear materialize all data assets...")
        isntantiated_data_assets_list = self.get_instantiated_data_assets_list()
        total_data_assets = len(isntantiated_data_assets_list)
        for idx, data_asset in enumerate(isntantiated_data_assets_list):
            curr_item = idx + 1
            self.LOGGER.debug(f"[{curr_item:,}/{total_data_assets:,}] Materializing Data Asset {data_asset.ALIAS}...")
            data_asset.materialize()
            self.LOGGER.debug(f"[{curr_item:,}/{total_data_assets:,}] Materializing Data Asset {data_asset.ALIAS}... Done!")
        self.LOGGER.debug("Linear materialize all data assets... Done!")
    
    def run(self):
        """Execute the job with timing, structured logging, and error propagation.

        Logs start/end times and duration. Re-raises any exception after logging.
        """
        self.start_time = datetime.now()
        self.LOGGER.info(f"Starting Job {self.ALIAS} at {self.start_time}")
        try:
            self.execute_on_run()
        except Exception as e:
            self.LOGGER.error(f"Job {self.ALIAS} failed: {e}")
            self.LOGGER.error(traceback.format_exc())
            raise e
        finally:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            self.LOGGER.info(f"Finished Job {self.ALIAS} at {self.end_time} (duration: {duration:.2f}s)")
            
            
class Etl(Job):
    """Job that structures execution into explicit extract, transform, and load phases.

    Subclasses must implement extract(), transform(), and load().
    Results of each phase are stored in self.extract_response, self.transform_response,
    and self.load_response respectively.
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'DATA_ASSETS',
        'TRANSFORMER_ASSETS',
        'LOAD_ASSETS',
    ]
    RESOURCES: List[Resource] = []
    DATA_ASSETS: List[DataAsset] = []
    TRANSFORMER_ASSETS: List[TransformerAsset] = []
    LOAD_ASSETS: List[LoadAsset] = []
    
    def extract(self) -> Any:
        """Extract raw data from sources. Must be implemented by subclasses."""
        raise NotImplementedError

    def transform(self) -> Any:
        """Transform extracted data. Must be implemented by subclasses."""
        raise NotImplementedError

    def load(self) -> Any:
        """Load transformed data to the destination. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def execute_on_run(self):
        self.LOGGER.info("Extracting...")
        dt0 = datetime.now()
        self.extract_response = self.extract()
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.info(f"Extract done in {duration:.2f}s... Done!")
        self.LOGGER.info("Transforming...")
        dt0 = datetime.now()
        self.transform_response = self.transform()
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.info(f"Transform done in {duration:.2f}s... Done!")
        self.LOGGER.info("Loading...")
        dt0 = datetime.now()
        self.load_response = self.load()
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.info(f"Load done in {duration:.2f}... Done!")