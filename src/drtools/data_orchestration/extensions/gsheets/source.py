


from typing import List
import pandas as pd
from ...main.source import TabularSource
from .resource import BaseGsheetsResource


class GoogleSheetsSource(TabularSource):
    """TabularSource backed by a specific Google Sheets worksheet.

    Class attributes:
        GSHEETS_ID: Google Sheets document ID. Must be set by subclasses.
        SHEET: Worksheet tab name. Must be set by subclasses.

    Raises:
        Exception: On init if GSHEETS_ID or SHEET are not set.
    """

    GSHEETS_ID = None
    SHEET = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.GSHEETS_ID:
            raise Exception("Static attribute GSHEETS_ID must be set.")
        if not self.SHEET:
            raise Exception("Static attribute SHEET must be set.")

    def _get_gsheets_resource(self, resources: List) -> BaseGsheetsResource:
        """Return the first BaseGsheetsResource from the provided list.

        Raises:
            Exception: If no BaseGsheetsResource is found.
        """
        resource = next((r for r in (resources or []) if isinstance(r, BaseGsheetsResource)), None)
        if resource is None:
            raise Exception(
                f"{self.ALIAS}: no BaseGsheetsResource found in provided resources. "
                "Make sure the calling Asset declares one in RESOURCES."
            )
        return resource

    def fetch(self, resources=None) -> pd.DataFrame:
        """Read the configured worksheet and return it as a DataFrame.

        Args:
            resources: List of instantiated resources from the calling Asset.
                Must contain at least one BaseGsheetsResource.

        Returns:
            DataFrame with columns matching this source's schema.
        """
        gsheets_resource = self._get_gsheets_resource(resources)
        data_df = gsheets_resource.sheet_as_df(gsheet_id=self.GSHEETS_ID, sheet=self.SHEET)
        return data_df[self.list_all_column_names()]

    def push(self, data: pd.DataFrame, resources=None) -> dict:
        """Write a DataFrame to the configured worksheet, replacing all existing content.

        Clears the sheet before writing so no stale rows are left behind.

        Args:
            data: DataFrame whose columns match this source's schema.
            resources: List of instantiated resources from the calling Asset.
                Must contain at least one BaseGsheetsResource.

        Returns:
            The gspread update response dict.
        """
        gsheets_resource = self._get_gsheets_resource(resources)
        columns = self.list_all_column_names()
        rows = data[columns].fillna('').astype(str).values.tolist()
        matrix = [columns] + rows
        gsheets_resource.clear_sheet(self.GSHEETS_ID, self.SHEET)
        return gsheets_resource.update_sheet(matrix, self.GSHEETS_ID, self.SHEET, 'A1', raw=False)