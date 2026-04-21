

from typing import List, Dict, Tuple
from datetime import datetime
from .types import ArgsKwargs
from copy import deepcopy
from .common import BaseAttributesHandler, ContextComponent
from .resource import Resource
from .source import Source


class AssetCheckResult:
    """Result of an asset check execution.

    Attributes:
        passed: True if the check succeeded, False otherwise.
        metadata: Optional dict with additional context (e.g. error message).
    """

    def __init__(self, passed: bool, metadata: Dict=None):
        self.passed = passed
        self.metadata = metadata


class CheckAsset(BaseAttributesHandler):
    """Base class for asset validation checks, executed before or after materialization.

    Subclasses must implement check() to define the validation logic.
    The asset being validated is injected via set_asset() before check() is called.
    """
    
    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'DATA_ASSETS',
    ]
    RESOURCES: List[ContextComponent] = []
    DATA_ASSETS: List = []
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if self.DESCRIPTION is None:
        #     raise Exception("Static attribute DESCRIPTION must be set.")
        self.asset = None
    
    def set_asset(self, asset):
        """Inject the asset being checked, making it available as self.asset.

        Args:
            asset: The Asset instance currently being validated.
        """
        self.asset = asset

    def check(self, *args, **kwargs) -> AssetCheckResult:
        """Execute the check logic. Must be implemented by subclasses.

        Returns:
            AssetCheckResult indicating whether the check passed.
        """
        raise NotImplementedError


class Asset(BaseAttributesHandler):
    """Base class for all pipeline assets.

    Orchestrates pre-check → execute → post-check lifecycle and measures execution time.

    Args:
        pre_check_asset: If True, pre-check assets are executed before materialization.
        post_check_asset: If True, post-check assets are executed after materialization.
    """
    
    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'PRE_CHECK_ASSETS',
        'POST_CHECK_ASSETS',
    ]
    RESOURCES: List[Resource] = []
    PRE_CHECK_ASSETS: List[CheckAsset] = []
    POST_CHECK_ASSETS: List[CheckAsset] = []
    
    def __init__(
        self, 
        pre_check_asset: bool=True,
        post_check_asset: bool=True,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pre_check_asset = pre_check_asset
        self.post_check_asset = post_check_asset
        self.materialize_response = None
    
    def perform_check_asset(self, *args, asset, instantiated_check_asset_list: List, **kwargs):
        """Run all checks from the given list sequentially, raising on first failure.

        Args:
            asset: The asset being checked.
            instantiated_check_asset_list: Ordered list of CheckAsset instances to run.

        Raises:
            Exception: If any check does not pass.
        """
        total_checks = len(instantiated_check_asset_list)
        for idx, check_asset in enumerate(instantiated_check_asset_list):
            curr_item = idx+1
            self.LOGGER.debug(f"[{curr_item:,}/{total_checks:,}] Checking asset {check_asset.ALIAS}...")
            check_asset.set_asset(asset)
            asset_check_result = check_asset.check(*args, **kwargs)
            if not asset_check_result.passed:
                msg = f"Asset {self.ALIAS} not passed on Pre Check Asset {check_asset.ALIAS}."
                self.LOGGER.error(msg)
                self.LOGGER.error(f"Metadata: {asset_check_result.metadata}")
                raise Exception(msg)
            self.LOGGER.debug(f"[{curr_item:,}/{total_checks:,}] Checking asset {check_asset.ALIAS}... Done!")
    
    def perform_pre_check_asset(self, asset, args_kwargs: ArgsKwargs=None):
        """Execute all registered PRE_CHECK_ASSETS.

        Args:
            asset: The asset being checked.
            args_kwargs: Optional ArgsKwargs forwarded to each check.
        """
        check_asset_list = self.get_instantiated_pre_check_assets_list()
        check_result = None
        if check_asset_list:
            self.LOGGER.debug("Pre checking assets...")
            if args_kwargs is None:
                args_kwargs = {}
            check_result = self.perform_check_asset(
                *deepcopy(args_kwargs.get('args', ())), 
                asset=asset,
                instantiated_check_asset_list=check_asset_list,
                **deepcopy(args_kwargs.get('kwargs', {})), 
            )
            self.LOGGER.debug("Pre checking assets... Done!")
        return check_result
    
    def perform_post_check_asset(self, asset, args_kwargs: ArgsKwargs=None):
        """Execute all registered POST_CHECK_ASSETS.

        Args:
            asset: The asset being checked.
            args_kwargs: Optional ArgsKwargs forwarded to each check.
        """
        check_asset_list = self.get_instantiated_post_check_assets_list()
        check_result = None
        if check_asset_list:
            self.LOGGER.debug("Post checking assets...")
            if args_kwargs is None:
                args_kwargs = {}
            check_result = self.perform_check_asset(
                *deepcopy(args_kwargs.get('args', ())), 
                asset=asset,
                instantiated_check_asset_list=check_asset_list,
                **deepcopy(args_kwargs.get('kwargs', {})), 
            )
            # check_result = self.perform_check_asset(deepcopy(asset_data), self.get_instantiated_post_check_assets_list())
            self.LOGGER.debug("Post checking assets... Done!")
        return check_result
    
    def get_materialize_response(self):
        """Return a deep copy of the last materialization result."""
        return deepcopy(self.materialize_response)

    def execute_on_materialize(self, *args, **kwargs):
        """Core materialization logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def materialize(self, *args, **kwargs):
        """Execute the full materialization lifecycle with logging and timing.

        Calls execute_on_materialize() and stores the result in self.materialize_response.

        Returns:
            The result of execute_on_materialize().
        """
        self.LOGGER.debug(f"Materializing {self.get_base_and_alias()}...")
        dt0 = datetime.now()
        materialize_response = self.execute_on_materialize(*args, **kwargs)
        self.materialize_response = deepcopy(materialize_response)
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.debug(f"Materialize {self.get_base_and_alias()} done in {duration:.2f}s... Done!")
        return materialize_response


class DataAsset(Asset):
    """Asset responsible for ingesting data from a source.

    Class attributes:
        SOURCE: Optional TabularSource class. If set, it is instantiated on __init__.

    Lifecycle: ingest() → post-check (if enabled).
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'DATA_ASSETS',
        'POST_CHECK_ASSETS',
    ]
    RESOURCES: List[Resource] = []
    DATA_ASSETS: List = []
    POST_CHECK_ASSETS: List[CheckAsset] = []
    SOURCE: Source = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if not self.SOURCE:
        #     raise Exception("Static attribute SOURCE must be set.")
        if self.SOURCE:
            # print("instantiate source")
            self.SOURCE = self.SOURCE(conf=self.context.conf, LOGGER=self.LOGGER)
        self.ingested_data = None
    
    def get_ingested_data(self):
        """Return a deep copy of the last ingested data."""
        return deepcopy(self.ingested_data)

    def execute_on_materialize(self):
        ingested_data = self.ingest()
        self.ingested_data = deepcopy(ingested_data)
        if self.post_check_asset:
            self.perform_post_check_asset(self, {'args': (ingested_data,)})
        return ingested_data

    def ingest(self):
        """Fetch and return raw data. Must be implemented by subclasses."""
        raise NotImplementedError


class TransformerAsset(Asset):
    """Asset responsible for transforming data through pre_transform → transform → post_transform.

    Lifecycle: pre-check → pre_transform → transform → post_transform → post-check.
    Results of each stage are stored and accessible via getters.
    """
    
    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'FROM_SOURCES',
        'RESOURCES',
        'PRE_CHECK_ASSETS',
        'POST_CHECK_ASSETS',
        'TO_SOURCES',
    ]
    FROM_SOURCES: List[Source] = []
    RESOURCES: List[Resource] = []
    PRE_CHECK_ASSETS: List[CheckAsset] = []
    POST_CHECK_ASSETS: List[CheckAsset] = []
    TO_SOURCES: List[Source] = []
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if not self.FROM_SOURCES:
        #     raise Exception(f"Static attribute FROM_SOURCES must be set on {self.ALIAS}.")
        # if not self.TO_SOURCES:
        #     raise Exception(f"Static attribute TO_SOURCES must be set on {self.ALIAS}.")
        self.transform_args = ()
        self.transform_kwargs = {}
        self.pre_transform_response = None
        self.transform_response = None
        self.post_transform_response = None
    
    def set_transform_args(self, args):
        """Store positional arguments to be used during transformation."""
        self.transform_args = deepcopy(args)

    def set_transform_kwargs(self, kwargs):
        """Store keyword arguments to be used during transformation."""
        self.transform_kwargs = deepcopy(kwargs)

    def get_transform_args(self) -> Tuple:
        """Return a deep copy of stored positional transform arguments."""
        return deepcopy(self.transform_args)

    def get_transform_kwargs(self) -> Dict:
        """Return a deep copy of stored keyword transform arguments."""
        return deepcopy(self.transform_kwargs)

    def get_transform_args_kwargs(self) -> ArgsKwargs:
        """Return stored args and kwargs packaged as an ArgsKwargs dict."""
        return ArgsKwargs(
            args=self.get_transform_args(),
            kwargs=self.get_transform_kwargs(),
        )

    def get_pre_transform_response(self):
        """Return a deep copy of the pre_transform() result."""
        return deepcopy(self.pre_transform_response)

    def get_transform_response(self):
        """Return a deep copy of the transform() result."""
        return deepcopy(self.transform_response)

    def get_post_transform_response(self):
        """Return a deep copy of the post_transform() result."""
        return deepcopy(self.post_transform_response)
    
    def execute_on_materialize(self, *args, **kwargs):
        self.set_transform_args(args)
        self.set_transform_kwargs(kwargs)
        
        if self.pre_check_asset:
            self.perform_pre_check_asset(self, self.get_transform_args_kwargs())
            
        self.LOGGER.debug("Pre transforming...")
        dt0 = datetime.now()
        pre_transform_response = self.pre_transform(
            *self.get_transform_args(),
            **self.get_transform_kwargs(),
        )
        self.pre_transform_response = deepcopy(pre_transform_response)
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.debug(f"Pre transform done in {duration:.2f}s... Done!")
        
        self.LOGGER.debug("Transforming...")
        dt0 = datetime.now()
        transform_response = self.transform(
            *self.get_pre_transform_response().get('args', ()), 
            **self.get_pre_transform_response().get('kwargs', {})
        )
        self.transform_response = deepcopy(transform_response)
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.debug(f"Transform done in {duration:.2f}s... Done!")
        
        self.LOGGER.debug("Post transforming...")
        dt0 = datetime.now()
        post_transform_response = self.post_transform(self.get_transform_response())
        self.post_transform_response = deepcopy(post_transform_response)
        duration = (datetime.now() - dt0).total_seconds()
        self.LOGGER.debug(f"Post transform done in {duration:.2f}s... Done!")
        
        if self.post_check_asset:
            self.perform_post_check_asset(self, {'args': (self.get_post_transform_response(),)})
            
        return post_transform_response
    
    def pre_transform(self, *args, **kwargs) -> ArgsKwargs:
        """Prepare args/kwargs before transform(). Default implementation passes them through.

        Returns:
            ArgsKwargs to be unpacked and forwarded to transform().
        """
        return ArgsKwargs(args=args, kwargs=kwargs)

    def transform(self, *args, **kwargs):
        """Apply the main transformation logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def post_transform(self, transform_response):
        """Process the transform result. Default implementation passes it through.

        Args:
            transform_response: The value returned by transform().

        Returns:
            The (optionally modified) transform response.
        """
        return transform_response


class LoadAsset(Asset):
    """Asset responsible for loading transformed data to a destination.

    Requires PRE_CHECK_ASSETS to be set, ensuring data is validated before loading.

    Class attributes:
        SOURCE: Optional TabularSource class. If set, it is instantiated on __init__.

    Lifecycle: pre-check (mandatory) → load().

    Raises:
        Exception: On init if PRE_CHECK_ASSETS is empty.
    """
    
    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'DATA_ASSETS',
        'PRE_CHECK_ASSETS',
        'POST_CHECK_ASSETS',
    ]
    RESOURCES: List[Resource] = []
    DATA_ASSETS: List = []
    PRE_CHECK_ASSETS: List[CheckAsset] = []
    POST_CHECK_ASSETS: List[CheckAsset] = []
    SOURCE: Source = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.PRE_CHECK_ASSETS:
            raise Exception("Static attribute PRE_CHECK_ASSETS must be set.")
        if self.SOURCE:
            self.SOURCE = self.SOURCE(conf=self.context.conf, LOGGER=self.LOGGER)
        self.load_response = None
    
    def get_load_response(self):
        """Return a deep copy of the last load() result."""
        return deepcopy(self.load_response)

    def execute_on_materialize(self, data):
        self.perform_pre_check_asset(self, {'args': (data,)})
        load_response = self.load(deepcopy(data))
        self.load_response = deepcopy(load_response)
        return load_response

    def load(self, data):
        """Send data to the destination. Must be implemented by subclasses.

        Args:
            data: Validated data to be loaded.
        """
        raise NotImplementedError