

from .common import BaseAttributesHandler
from typing import List
from .resource import Resource
from .job import Job


class Workflow(BaseAttributesHandler):
    """Base class for orchestrating multiple jobs.

    Subclasses must implement run() to define execution order.
    """

    ATTR_NAMES: List[str] = [
        'RESOURCES',
        'JOBS',
    ]
    RESOURCES: List[Resource] = []
    JOBS: List[Job] = []

    def run(self):
        """Execute the workflow. Must be implemented by subclasses."""
        raise NotImplementedError


class LinearWorkflow(Workflow):
    """Workflow that runs all registered JOBS sequentially in declaration order.

    Each job is instantiated fresh with the workflow's conf and logger before running.
    """

    def run(self):
        for job in self.JOBS:
            job(conf=self.context.conf, LOGGER=self.LOGGER).run()