

from typing import List, Dict
from pandas import DataFrame
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from drtools.file_manager import load_json
from gspread.spreadsheet import Spreadsheet
from ...main.resource import SingletonResource
import pandas as pd


class BaseGsheetsResource(SingletonResource):
    """Singleton resource for interacting with Google Sheets via a service account.

    Class attributes:
        SERVICE_ACCOUNT_CREDENTIALS: Path to the service account JSON credentials file.
        SCOPES: List of Google API OAuth2 scopes. Defaults to spreadsheets read/write.

    Raises:
        Exception: On init if SERVICE_ACCOUNT_CREDENTIALS or SCOPES are not set.
    """

    NAME: str = "Gsheets Resource"
    SERVICE_ACCOUNT_CREDENTIALS: str = None
    SCOPES: List[str] = [
        'https://www.googleapis.com/auth/spreadsheets',
        # 'https://www.googleapis.com/auth/drive',
    ]

    @staticmethod
    def load_sheet_as_df(
        spreadsheet: Spreadsheet,
        name: str,
    ) -> DataFrame:
        """Load a worksheet from an open Spreadsheet object as a DataFrame.

        Args:
            spreadsheet: An open gspread Spreadsheet instance.
            name: Worksheet tab name.

        Returns:
            DataFrame with the first row as column headers and remaining rows as data.
        """
        worksheet = spreadsheet.worksheet(name)
        all_values = worksheet.get_all_values()
        gsheets_data = []
        if len(all_values) > 1:
            gsheets_data = all_values[1:]
        df = pd.DataFrame(gsheets_data, columns=all_values[0])
        return df
    
    def __init__(
        self, 
        conf: Dict=None,
        LOGGER = None
    ):
        self.GOOGLE_CLIENT = None
        super().__init__(conf, LOGGER)
        if not self.SERVICE_ACCOUNT_CREDENTIALS:
            raise Exception("Static attribute SERVICE_ACCOUNT_CREDENTIALS must be set.")
        if not self.SCOPES:
            raise Exception("Static attribute SCOPES must be set.")
        self.credentials_data = load_json(self.SERVICE_ACCOUNT_CREDENTIALS)
        
    def login_to_google_api(self):
        """Authenticate with the Google API using service account credentials.

        Stores the authorized client in self.GOOGLE_CLIENT.
        """
        self.LOGGER.debug('Login to Google API...')
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            self.credentials_data, 
            scopes=self.SCOPES
        )
        self.GOOGLE_CLIENT = gspread.authorize(credentials)
        self.LOGGER.debug('Login to Google API... Done!')
    
    @property
    def google_client(self):
        """Return the authorized Google API client, authenticating lazily on first access."""
        if not self.GOOGLE_CLIENT:
            self.login_to_google_api()
        return self.GOOGLE_CLIENT
    
    def spreadsheet(
        self,
        gsheet_id: str,
    ) -> Spreadsheet:
        """Open and return a Spreadsheet object by its Google Sheets ID.

        Args:
            gsheet_id: The Google Sheets document ID.
        """
        spreadsheet: Spreadsheet = self.google_client.open_by_key(gsheet_id)
        return spreadsheet
    
    def sheet_as_df(
        self,
        gsheet_id: str,
        sheet: str
    ) -> DataFrame:
        """Fetch a worksheet as a DataFrame.

        Args:
            gsheet_id: The Google Sheets document ID.
            sheet: Worksheet tab name.
        """
        spreadsheet = self.spreadsheet(gsheet_id)
        return self.load_sheet_as_df(spreadsheet, sheet)
    
    def update_sheet(
        self,
        data,
        gsheet_id: str,
        sheet: str,
        cell_start: str,
        **kwargs,
    ) -> Dict:
        """Write data to a worksheet starting at a given cell.

        Args:
            data: Data to write (list of lists).
            gsheet_id: The Google Sheets document ID.
            sheet: Worksheet tab name.
            cell_start: Top-left cell reference (e.g. 'A1', 'B2').
            **kwargs: Additional kwargs forwarded to gspread worksheet.update().

        Returns:
            The gspread update response dict.
        """
        spreadsheet = self.spreadsheet(gsheet_id)
        worksheet = spreadsheet.worksheet(sheet)
        update_reponse = worksheet.update(data, cell_start, **kwargs)
        return update_reponse

    def clear_sheet(self, gsheet_id: str, sheet: str) -> Dict:
        """Clear all content from a worksheet.

        Args:
            gsheet_id: The Google Sheets document ID.
            sheet: Worksheet tab name.

        Returns:
            The gspread clear response dict.
        """
        spreadsheet = self.spreadsheet(gsheet_id)
        worksheet = spreadsheet.worksheet(sheet)
        return worksheet.clear()
    