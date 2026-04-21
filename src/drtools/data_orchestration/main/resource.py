

from .common import ContextComponent


class Resource(ContextComponent):
    """Base class for all pipeline resources.

    A resource represents an external dependency (e.g. a database connection,
    an API client, a browser session) that can be shared across pipeline components.
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None


class SingletonResource(Resource):
    """Resource that ensures only one instance exists per subclass (Singleton pattern).

    Useful for resources that are expensive to initialize (e.g. API clients,
    browser sessions) and should be shared across all components in a pipeline run.
    """

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None

    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # create just one time
            cls._instance = super().__new__(cls)
        return cls._instance