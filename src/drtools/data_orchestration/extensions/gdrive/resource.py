

from typing import List, Dict
from drtools.types import JSONLike
from drtools.google.drive.drive import DriveFromServiceAcountFile
from datetime import datetime, timedelta
from drtools.utils import ProgressETA
from pandas import DataFrame
import pandas as pd
from ...main.resource import SingletonResource
import pandas as pd


class BaseGdriveResource(SingletonResource):
    """Singleton resource for interacting with Google Drive via a service account.

    Class attributes:
        DRIVE_SERVICE_CREDENTIALS: Path to the service account JSON credentials file.

    Raises:
        Exception: On init if DRIVE_SERVICE_CREDENTIALS is not set.
    """

    NAME: str = "Gdrive Resource"
    DRIVE = None
    DRIVE_SERVICE_CREDENTIALS: str = None

    def __init__(
        self,
        conf: Dict=None,
        LOGGER = None
    ):
        super().__init__(conf, LOGGER)
        if not self.DRIVE_SERVICE_CREDENTIALS:
            raise Exception("Static attribute DRIVE_SERVICE_CREDENTIALS must be set.")

    def build_drive(self):
        """Instantiate and build the Drive client from the service account credentials file.

        Stores the client in self.DRIVE.
        """
        self.LOGGER.debug("Building Gdrive...")
        self.DRIVE = DriveFromServiceAcountFile(self.DRIVE_SERVICE_CREDENTIALS, LOGGER=self.LOGGER)
        self.DRIVE.build()
        self.LOGGER.debug("Building Gdrive... Done!")
        
    @property
    def drive(self):
        """Return the Drive client, building it lazily on first access."""
        if not self.DRIVE:
            self.build_drive()
        return self.DRIVE
    
    def get_file_or_last_modified_file_content_from_folder(
        self,
        folder: str,
        filename: str=None
    ) -> JSONLike:
        """Return the content of a specific file or the last modified file in a folder.

        Args:
            folder: Drive folder path.
            filename: If provided, fetch this specific file. Otherwise fetch the
                most recently modified file in the folder.

        Returns:
            Parsed file content as a JSON-like structure.
        """
        if filename:
            raw_data = self.drive.get_file_content_from_path(f'{folder}/{filename}')
        else:
            raw_data = self.drive.get_last_modified_file_content_from_folder(folder)
        return raw_data
    
    def csv_files_content_from_folder(
        self,
        folder: str,
        start_date: str=None, # Default: now() - 7 days
        end_date: str=None,
        **get_file_content_kwargs
    ) -> List[Dict]:
        """Download all files from a Drive folder modified within a date range.

        Args:
            folder: Drive folder path to search.
            start_date: ISO date string ('YYYY-MM-DD') for the start of the range.
                Defaults to 7 days ago.
            end_date: ISO date string ('YYYY-MM-DD') for the end of the range.
                Defaults to 2 days from now.
            **get_file_content_kwargs: Extra kwargs forwarded to the Drive get_file_content call.

        Returns:
            List of dicts, each with 'file_metadata' (Drive metadata) and 'content' (parsed file data).
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        self.LOGGER.debug(f'Finding files from {start_date} to {end_date}')
        response = self.drive.list_files_from_folder_filtering_by_modified_time(
            folder,
            start_date=start_date,
            end_date=end_date,
        )
        self.LOGGER.debug(f'Found {len(response["files"]):,} files.')
        self.LOGGER.debug(f'Downloading...')
        data = []
        for r in ProgressETA(response['files'], LOGGER=self.LOGGER):
            self.LOGGER.debug(f'File name: {r["name"]}')
            dict_data = self.drive.get_file_content(r['id'], mimetype=r['mimeType'], **get_file_content_kwargs)
            data.append({
                'file_metadata': r,
                'content': dict_data
            })
        self.LOGGER.debug(f'Downloading... Done!')
        return data
    
    @classmethod
    def csv_files_content_to_df(
        cls,
        data: List[Dict]
    ) -> DataFrame:
        """Flatten the output of csv_files_content_from_folder into a single DataFrame.

        Each row in the result combines a content record with its file metadata columns:
        file_id, file_name, file_mimetype, file_created_time, file_modified_time, file_kind.

        Args:
            data: List returned by csv_files_content_from_folder.

        Returns:
            DataFrame with one row per content record, enriched with file metadata.
        """
        real_data = []
        for item in data:
            for r in item['content']:
                real_data.append({
                    **r,
                    'file_id': item['file_metadata']['id'], 
                    'file_name': item['file_metadata']['name'], 
                    'file_mimetype': item['file_metadata']['mimeType'], 
                    'file_created_time': item['file_metadata']['createdTime'], 
                    'file_modified_time': item['file_metadata']['modifiedTime'], 
                    'file_kind': item['file_metadata']['kind'], 
                })
        data_df = pd.DataFrame(real_data)
        return data_df
        
    
    def csv_files_content_from_folder_as_df(
        self,
        folder: str,
        start_date: str=None, # Default: now() - 7 days
        end_date: str=None,
        **get_file_content_kwargs
    ) -> DataFrame:
        """Convenience method: download files from a folder and return them as a DataFrame.

        Combines csv_files_content_from_folder and csv_files_content_to_df.

        Args:
            folder: Drive folder path to search.
            start_date: ISO date string for the start of the range. Defaults to 7 days ago.
            end_date: ISO date string for the end of the range. Defaults to 2 days from now.
            **get_file_content_kwargs: Extra kwargs forwarded to the Drive get_file_content call.

        Returns:
            DataFrame with all file content rows enriched with file metadata.
        """
        data = self.csv_files_content_from_folder(
            folder=folder,
            start_date=start_date,
            end_date=end_date,
            **get_file_content_kwargs
        )
        data_df = self.csv_files_content_to_df(data)
        return data_df