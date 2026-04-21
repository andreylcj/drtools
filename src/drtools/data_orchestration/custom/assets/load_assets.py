

from ...main.assets import LoadAsset
from ...main.source import TabularSource
from .check_assets import (
    CheckDataIsMatrix,
    CheckManualColumnsFromMatrixData,
    ValidateDataFromMatrix
)


class TabularLoadAsset(LoadAsset):
    """LoadAsset that requires a TabularSource to be set.

    Validates on init that SOURCE is present and is an instance of TabularSource.
    """

    SOURCE: TabularSource = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.SOURCE:
            raise Exception("Statice attribute SOURCE must be set.")
        if not isinstance(self.SOURCE, TabularSource):
            raise Exception("Statice attribute SOURCE must be a TabularSource instance.")


class TabularAsMatrixLoadAsset(TabularLoadAsset):
    """TabularLoadAsset that automatically registers matrix validation pre-checks.

    On init, prepends CheckDataIsMatrix, CheckManualColumnsFromMatrixData, and
    ValidateDataFromMatrix to PRE_CHECK_ASSETS.
    """

    SOURCE: TabularSource = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PRE_CHECK_ASSETS = [
                CheckDataIsMatrix, 
                CheckManualColumnsFromMatrixData,
                ValidateDataFromMatrix
            ] + self.PRE_CHECK_ASSETS