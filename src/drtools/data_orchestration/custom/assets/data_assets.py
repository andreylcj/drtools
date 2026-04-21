

from ...main.assets import DataAsset
from ...main.source import TabularSource
from .check_assets import (
    CheckDataIsDataframe,
    CheckAllColumnsFromDataframe,
    ValidateDataFromDataframe
)


class TabularDataAsset(DataAsset):
    """DataAsset that requires a TabularSource to be set.

    Validates on init that SOURCE is present and is an instance of TabularSource.
    """

    SOURCE: TabularSource = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.SOURCE:
            raise Exception("Statice attribute SOURCE must be set.")
        if not isinstance(self.SOURCE, TabularSource):
            raise Exception("Statice attribute SOURCE must be a TabularSource instance.")


class TabularAsDataframeDataAsset(TabularDataAsset):
    """TabularDataAsset that automatically registers DataFrame validation post-checks.

    On init, prepends CheckDataIsDataframe, CheckAllColumnsFromDataframe, and
    ValidateDataFromDataframe to POST_CHECK_ASSETS.
    """

    SOURCE: TabularSource = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.POST_CHECK_ASSETS = [
            CheckDataIsDataframe, 
            CheckAllColumnsFromDataframe,
            ValidateDataFromDataframe,
        ] + self.POST_CHECK_ASSETS