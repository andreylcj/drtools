

from drtools.logging import Logger, FormatterOptions
from typing import List, Dict
from datetime import datetime
from .utils import has_unique_elements, get_duplicates


class Context:
    """Holds runtime configuration and reference date for a pipeline execution.

    Attributes:
        conf: Arbitrary configuration dict accessible to all pipeline components.
        start_date: Reference datetime for the pipeline run. Defaults to now().
    """

    def __init__(
        self,
        conf: Dict=None,
        start_date: str=None, # YYYY-MM-DD
    ):
        """Initialize the context.

        Args:
            conf: Optional configuration dict.
            start_date: Optional ISO-format date string ('YYYY-MM-DD'). Defaults to now().
        """
        if not conf:
            conf = {}
        self.conf = conf
        if start_date is None:
            start_date = datetime.now()
        else:
            start_date = datetime.fromisoformat(start_date)
        self.start_date = start_date
            
            
class ContextComponent:
    """Base class for all pipeline components (resources, assets, sources, jobs).

    Provides a shared Logger and Context, and utility methods for component identification.

    Class attributes:
        NAME: Human-readable name for the component.
        DESCRIPTION: Short description of what the component does.
        ALIAS: Identifier used internally for lookups. Defaults to the class name.
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get("ALIAS", None) is None:
            cls.ALIAS = cls.__name__

    def __init__(self, conf: Dict=None, LOGGER: Logger=None):
        """Initialize the component.

        Args:
            conf: Optional configuration dict forwarded to Context.
            LOGGER: Optional Logger instance. A default logger is created if not provided.
        """
        # if self.NAME is None:
        #     raise Exception(f"Static attribute NAME must be set on {self.__class__.__name__}.")
        if LOGGER is None:
            LOGGER = Logger(
                name=self.ALIAS,
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_thread_name=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            )
        self.LOGGER = LOGGER
        self.context = Context(conf)
    
    def get_base(self) -> str:
        """Return the immediate parent class name of this component."""
        return str(self.__class__.__base__).split('.')[-1][:-2]

    def get_base_and_name(self) -> str:
        """Return a string combining the parent class name and this class name."""
        return f"{self.get_base()} {self.__class__.__name__}"

    def get_base_and_alias(self) -> str:
        """Return a string combining the parent class name and this component's ALIAS."""
        return f"{self.get_base()} {self.ALIAS}"

    def get_rel_dt(self) -> datetime:
        """Return the reference datetime for this execution.

        Checks context.conf['start_date'] first; falls back to context.start_date.
        """
        rel_dt = self.context.start_date
        if self.context.conf.get('start_date', None):
            rel_dt = datetime.fromisoformat(self.context.conf['start_date'])
        return rel_dt
    


class AttributeDefaultEmptyClass: 
    pass


class AttributesAsListOfUniqueContextComponentClassesNotInstantiatedHandler(ContextComponent):
    """Manages lists of ContextComponent subclasses, instantiating them on initialization.

    Subclasses declare ATTR_NAMES listing class-level attributes, each being a list
    of uninstantiated ContextComponent subclasses. On __init__, each list is validated
    for uniqueness and then instantiated, storing results in '_INSTANTIATED_{ATTR_NAME}'.

    Class attributes:
        ATTR_NAMES: List of attribute names to process.
    """

    ATTR_NAMES: List[str] = None

    @staticmethod
    def construct_instantiated_name_from_attr_name(attr_name: str) -> str:
        """Return the name of the instantiated attribute for a given attr_name.

        Example: 'RESOURCES' -> '_INSTANTIATED_RESOURCES'
        """
        return f'_INSTANTIATED_{attr_name}'

    @staticmethod
    def construct_name_for_class(class_def) -> str:
        """Return the ALIAS of a ContextComponent class, used as its lookup key."""
        return class_def.ALIAS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ATTR_NAMES is None:
            raise Exception("Static attribute ATTR_NAMES must be set.")
        self._valid_classes_name = {}
        # self._attr_classes_alias = {}
        for attr_name in self.ATTR_NAMES:
            self._valid_classes_name[attr_name] = []
            self.handle_attr(attr_name)
    
    def validate_uniqueness(self, attr_name: str):
        """Raise if the list at attr_name contains classes with duplicate ALIAS values.

        Args:
            attr_name: Name of the class-level list attribute to validate.

        Raises:
            Exception: If duplicate ALIAS values are found.
        """
        attr_value = getattr(self, attr_name, None)
        if attr_value:
            names = [self.construct_name_for_class(x) for x in attr_value]
            if not has_unique_elements(names):
                duplicated = get_duplicates(names)
                raise Exception(f"Static Attribute {attr_name} must have only unique names. Found duplicated: {duplicated}")
        
    def instantiate_classes(self, attr_name: str):
        """Instantiate all classes in the list at attr_name and store them.

        Creates a '_INSTANTIATED_{attr_name}' attribute holding an
        AttributeDefaultEmptyClass whose sub-attributes are the instantiated components,
        keyed by their ALIAS.

        Args:
            attr_name: Name of the class-level list attribute to instantiate.
        """
        attr_value: List = getattr(self, attr_name, None)
        instantiated_name = self.construct_instantiated_name_from_attr_name(attr_name)
        setattr(self, instantiated_name, AttributeDefaultEmptyClass())
        instantiated_attr = getattr(self, instantiated_name)
        set_values = attr_value
        if not set_values:
            set_values = []
        for _class in set_values:
            _class_item_name = self.construct_name_for_class(_class)
            setattr(instantiated_attr, _class_item_name, _class)
        setattr(instantiated_attr, 'valid_classes_name', self._valid_classes_name[attr_name])
        for idx, class_list_item in enumerate(attr_value):
            item_name = self.construct_name_for_class(class_list_item)
            setattr(
                getattr(self, instantiated_name), 
                item_name, 
                getattr(getattr(self, instantiated_name), item_name)(
                    conf=self.context.conf, 
                    LOGGER=self.LOGGER
                )
            )
            self._valid_classes_name[attr_name].append(item_name)
    
    def handle_attr(self, attr_name: str):
        """Validate and instantiate the classes list for the given attribute.

        Args:
            attr_name: Name of the class-level list attribute to handle.

        Raises:
            Exception: If the attribute does not exist on the class.
        """
        attr_value = getattr(self, attr_name, "__EMPTY__")
        if attr_value == '__EMPTY__':
            raise Exception(f"Attribute {attr_name} does not exists.")
        self.validate_uniqueness(attr_name)
        self.instantiate_classes(attr_name)
    
    def get_instantiated_attr(self, attr_name: str):
        """Return the instantiated attribute container for the given attr_name.

        Args:
            attr_name: One of the names in ATTR_NAMES.

        Returns:
            The AttributeDefaultEmptyClass instance holding the instantiated components.

        Raises:
            Exception: If the attribute was not found or not yet instantiated.
        """
        instantiated_attr_name = self.construct_instantiated_name_from_attr_name(attr_name)
        instantiated_attr_value = getattr(self, instantiated_attr_name, None)
        if not instantiated_attr_value:
            raise Exception(f"No instantiated attribute for {attr_name} was found. Valid attr: {self.ATTR_NAMES}")
        if not isinstance(instantiated_attr_value, AttributeDefaultEmptyClass):
            raise Exception(f"Attribute {attr_name} was not instantiated yet.")
        return instantiated_attr_value
    
    def get_instantiated_attr_classes_as_list(self, attr_name: str):
        """Return all instantiated components for an attribute as an ordered list.

        Args:
            attr_name: One of the names in ATTR_NAMES.

        Returns:
            List of instantiated ContextComponent instances.
        """
        instantiated_attr_value = self.get_instantiated_attr(attr_name)
        list_of_classes = [getattr(instantiated_attr_value, class_name) for class_name in self._valid_classes_name[attr_name]]
        return list_of_classes
    
    def get_instantiated_attr_class(self, attr_name: str, class_name: str):
        """Return a specific instantiated component by its ALIAS.

        Args:
            attr_name: One of the names in ATTR_NAMES.
            class_name: The ALIAS of the component to retrieve.

        Raises:
            Exception: If class_name is not registered under attr_name.
        """
        instantiated_attr_value = self.get_instantiated_attr(attr_name)
        if class_name not in self._valid_classes_name[attr_name]:
            raise Exception(f"Not found class {class_name} for attribute {attr_name}. Valid classes names: {self._valid_classes_name[attr_name]}")
        instantiated_class_item = getattr(instantiated_attr_value, class_name)
        return instantiated_class_item
    
    
class BaseAttributesHandler(AttributesAsListOfUniqueContextComponentClassesNotInstantiatedHandler):
    """Standard handler that manages the full set of pipeline component lists.

    Extends the base handler with typed accessors for RESOURCES, CHECK_ASSETS,
    DATA_ASSETS, TRANSFORMER_ASSETS, LOAD_ASSETS, JOBS, FROM_SOURCES, and TO_SOURCES.
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None
    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'PRE_CHECK_ASSETS',
        'POST_CHECK_ASSETS',
        'DATA_ASSETS',
        'TRANSFORMER_ASSETS',
        'LOAD_ASSETS',
        'JOBS',
        'FROM_SOURCES',
        'TO_SOURCES',
    ]
    RESOURCES: List[ContextComponent] = []
    PRE_CHECK_ASSETS: List[ContextComponent] = []
    POST_CHECK_ASSETS: List[ContextComponent] = []
    DATA_ASSETS: List[ContextComponent] = []
    TRANSFORMER_ASSETS: List[ContextComponent] = []
    LOAD_ASSETS: List[ContextComponent] = []
    JOBS: List[ContextComponent] = []
    FROM_SOURCES: List[ContextComponent] = []
    TO_SOURCES: List[ContextComponent] = []
    
    def get_resource(self, name: str):
        """Return the instantiated resource with the given ALIAS."""
        return self.get_instantiated_attr_class('RESOURCES', name)

    def get_instantiated_resources_list(self) -> List:
        """Return all instantiated resources as a list."""
        return self.get_instantiated_attr_classes_as_list('RESOURCES')

    def get_instantiated_pre_check_assets_list(self) -> List:
        """Return all instantiated pre-check assets as a list."""
        return self.get_instantiated_attr_classes_as_list('PRE_CHECK_ASSETS')

    def get_instantiated_post_check_assets_list(self) -> List:
        """Return all instantiated post-check assets as a list."""
        return self.get_instantiated_attr_classes_as_list('POST_CHECK_ASSETS')

    def get_instantiated_data_assets_list(self) -> List:
        """Return all instantiated data assets as a list."""
        return self.get_instantiated_attr_classes_as_list('DATA_ASSETS')

    def get_data_asset(self, name: str):
        """Return the instantiated data asset with the given ALIAS."""
        return self.get_instantiated_attr_class('DATA_ASSETS', name)

    def get_data_asset_ingested_data(self, name: str):
        """Return the ingested data from the data asset with the given ALIAS."""
        return self.get_data_asset(name).get_ingested_data()

    def get_instantiated_transformer_assets_list(self) -> List:
        """Return all instantiated transformer assets as a list."""
        return self.get_instantiated_attr_classes_as_list('TRANSFORMER_ASSETS')

    def get_transformer_asset(self, name: str):
        """Return the instantiated transformer asset with the given ALIAS."""
        return self.get_instantiated_attr_class('TRANSFORMER_ASSETS', name)

    def get_transformer_asset_pre_transform_response(self, name: str):
        """Return the pre-transform response from the transformer asset with the given ALIAS."""
        return self.get_transformer_asset(name).get_pre_transform_response()

    def get_transformer_asset_transform_response(self, name: str):
        """Return the transform response from the transformer asset with the given ALIAS."""
        return self.get_transformer_asset(name).get_transform_response()

    def get_transformer_asset_post_transform_response(self, name: str):
        """Return the post-transform response from the transformer asset with the given ALIAS."""
        return self.get_transformer_asset(name).get_post_transform_response()

    def get_instantiated_load_assets_list(self) -> List:
        """Return all instantiated load assets as a list."""
        return self.get_instantiated_attr_classes_as_list('LOAD_ASSETS')

    def get_load_asset(self, name: str):
        """Return the instantiated load asset with the given ALIAS."""
        return self.get_instantiated_attr_class('LOAD_ASSETS', name)

    def get_load_asset_load_response(self, name: str):
        """Return the load response from the load asset with the given ALIAS."""
        return self.get_load_asset(name).get_load_response()

    def get_instantiated_jobs_list(self) -> List:
        """Return all instantiated jobs as a list."""
        return self.get_instantiated_attr_classes_as_list('JOBS')

    def get_job(self, name: str):
        """Return the instantiated job with the given ALIAS."""
        return self.get_instantiated_attr_class('JOBS', name)

    def get_instantiated_from_sources_list(self) -> List:
        """Return all instantiated FROM sources as a list."""
        return self.get_instantiated_attr_classes_as_list('FROM_SOURCES')

    def get_from_source(self, name: str):
        """Return the instantiated FROM source with the given ALIAS."""
        return self.get_instantiated_attr_class('FROM_SOURCES', name)

    def get_instantiated_to_sources_list(self) -> List:
        """Return all instantiated TO sources as a list."""
        return self.get_instantiated_attr_classes_as_list('TO_SOURCES')

    def get_to_source(self, name: str):
        """Return the instantiated TO source with the given ALIAS."""
        return self.get_instantiated_attr_class('TO_SOURCES', name)