

from ...custom.assets.data_assets import TabularAsDataframeDataAsset
from .resource import BaseGsheetsResource
from .source import GoogleSheetsSource
from pandas import DataFrame


class GoogleSheetsDataAsset(TabularAsDataframeDataAsset):
    """DataAsset that ingests data from a Google Sheets worksheet as a DataFrame.

    Requires SOURCE to be a GoogleSheetsSource and RESOURCES to contain exactly
    one BaseGsheetsResource.

    Raises:
        Exception: On init if SOURCE is not a GoogleSheetsSource, RESOURCES is empty
            or contains more than one resource, or the resource is not a BaseGsheetsResource.
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
        self.GSHEETS_ID = self.SOURCE.GSHEETS_ID
        self.SHEET = self.SOURCE.SHEET

    def ingest(self) -> DataFrame:
        """Fetch the configured worksheet and return it as a DataFrame filtered to schema columns."""
        gsheets_resource = self.get_resource(self.RESOURCES[0].ALIAS)
        data_df = gsheets_resource.sheet_as_df(gsheet_id=self.GSHEETS_ID, sheet=self.SHEET)
        data_df = data_df[self.SOURCE.list_all_column_names()]
        return data_df