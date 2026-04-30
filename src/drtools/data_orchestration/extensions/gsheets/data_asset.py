

from ...custom.assets.data_assets import TabularAsDataframeDataAsset
from .resource import BaseGsheetsResource
from .source import GoogleSheetsSource


class GoogleSheetsDataAsset(TabularAsDataframeDataAsset):
    """DataAsset that ingests data from a Google Sheets worksheet as a DataFrame.

    Delegates fetching to GoogleSheetsSource.fetch() — no ingest() override needed.

    Requires SOURCE to be a GoogleSheetsSource and RESOURCES to contain exactly
    one BaseGsheetsResource.

    Raises:
        Exception: On init if SOURCE is not a GoogleSheetsSource, RESOURCES is empty,
            contains more than one resource, or the resource is not a BaseGsheetsResource.
    """

    SOURCE = None
    RESOURCES = [BaseGsheetsResource]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.SOURCE, GoogleSheetsSource):
            raise Exception("Source must be an instance of extensions.gsheets.source.GoogleSheetsSource.")
        if not self.RESOURCES:
            raise Exception("Static attribute RESOURCES must be set.")
        if len(self.RESOURCES) > 1:
            raise Exception("Static attribute RESOURCES must have just one Google Sheets Resource.")
        if not isinstance(self.get_resource(self.RESOURCES[0].ALIAS), BaseGsheetsResource):
            raise Exception("Resource must be instance of extensions.gsheets.resource.BaseGsheetsResource.")