

from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import List
from .types import (
    FileId,
    FilesListResult,
)
from .utils import (
    bytes_to_json
)
import io
from googleapiclient.http import MediaIoBaseDownload
from drtools.logging import Logger, FormatterOptions
from typing import Callable


class Drive:
    
    SCOPES: List[str] = ['https://www.googleapis.com/auth/drive']
    
    def __init__(
        self,
        credentials_method: Callable,
        *args,
        LOGGER: Logger=None,
        **kwargs,
    ) -> None:
        scopes = kwargs.pop('scopes', None)
        if not scopes:
            scopes = self.SCOPES
        self.credentials = credentials_method(*args, scopes=scopes, **kwargs)
        if not LOGGER:
            LOGGER = Logger(
                name="Drive",
                formatter_options=FormatterOptions(include_datetime=True, include_logger_name=True, include_level_name=True),
                default_start=False
            )
        self.LOGGER = LOGGER
        self.service = None
        
    def get_folders_from_name(
        self,
        name: str, 
        page_size: int=10,
        fields: str="nextPageToken, files(id, name, kind, mimeType)",
        parent_folder_id: str=None
    ) -> FilesListResult:
        """List folders and files in Google Drive."""
        q = f"mimeType='application/vnd.google-apps.folder' and name='{name}'"
        if parent_folder_id:
            q += f" and '{parent_folder_id}' in parents"
        results = self.service.files().list(q=q, pageSize=page_size, fields=fields).execute()
        return results
    
    def get_folder_content(
        self,
        folder_id, 
        page_size: int=1000, 
        trashed: bool=False, 
        fields: str="nextPageToken, files(id, name, kind, mimeType)",
        deep: bool=False,
    ) -> FilesListResult:
        """List folders and files in Google Drive."""
        def _get_folder_content(folder_id, page_size, trashed):
            trashed = 'true' if trashed else 'false'
            results = self.service.files().list(q=f"'{folder_id}' in parents and trashed={trashed}", pageSize=page_size, fields=fields).execute()
            items = results
            return items
        items = _get_folder_content(folder_id, page_size, trashed)
        if deep:
            items = [
                {**item, 'content': self.serive.get_folder_content(item['id'], page_size, trashed, True) if 'folder' in item['mimeType'] else None}
                for item in items
            ]
        return items

    def get_folder_content_from_path(
        self,
        path: str, 
        page_size: int=1000, 
        trashed: bool=False, 
        fields: str="nextPageToken, files(id, name, kind, mimeType)",
        deep: bool=False,
    ) -> FilesListResult:
        """List folders and files in Google Drive."""
        folder_names = path.split('/')
        parent_id = None
        parent_path = None
        for idx, folder_name in enumerate(folder_names):
            if idx == 0:
                results = self.get_folders_from_name(folder_name)
            else:
                results = self.get_folders_from_name(folder_name, parent_folder_id=parent_id)
            results_len = len(results['files'])
            if results_len > 1:
                if idx == 0:
                    raise Exception(f"Path root must be unique in Drive. Files results for {folder_name} were {results_len:,}.")
                else:
                    raise Exception(f"Folder name must be unique inside parent. Folder name: {folder_name} | Parent path: {parent_path} | Parent ID: {parent_id}")
            if results_len == 0:
                raise Exception(f"Folder not find. Folder name: {folder_name} | Parent path: {parent_path} | Parent ID: {parent_id}")
            parent_id = results['files'][0]['id']
            parent_path = folder_name if idx == 0 else f'{parent_path}/{folder_name}'
        return self.get_folder_content(parent_id, page_size, trashed, fields, deep)
    
    def create_folder(
        self,
        folder_name: str, 
        parent_folder_id: FileId, 
        ignore_if_exists: bool=True
    ) -> FileId:
        """Create a folder in Google Drive and return its ID."""
        folder_metadata = {
            'name': folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            'parents': [parent_folder_id] if parent_folder_id else []
        }
        if ignore_if_exists:
            items = self.get_folder_content(parent_folder_id)
            for item in items['files']:
                if folder_name == item['name']:
                    return None
        created_folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        return created_folder["id"]
    
    def get_file_content(
        self,
        file_id: str,
        try_handle_mime_type: bool=True,
        mime_type: str=None,
    ) -> bytes:
        request = self.service.files().get_media(fileId=file_id)
        # create_directories_of_path(filepath)
        fh = io.BytesIO()
        # fh = io.FileIO(filepath, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            self.LOGGER.debug(f"Download {int(status.progress() * 100)}%.")
        value = fh.getvalue()
        if try_handle_mime_type:
            value = self.handle_bytes_from_mime_type(value, mime_type)
        return value
    
    def get_file_content_from_path(
        self, 
        filepath: str,
        try_handle_mime_type: bool=True,
    ) -> io.BytesIO:
        folder_path = '/'.join(filepath.split('/')[:-1])
        filename = filepath.split('/')[-1]
        self_folder_content = self.get_folder_content_from_path(folder_path)
        file_item = [item for item in self_folder_content['files'] if item['name'] == filename]
        if len(file_item) > 1:
            raise Exception(f"There are more than 1 file with same path. Files founde: {len(file_item):,}")
        if len(file_item) == 0:
            raise Exception(f"No file found with path: {filepath}")
        file_id = file_item[0]['id']
        return self.get_file_content(file_id, try_handle_mime_type, file_item[0]['mimeType'])
    
    @classmethod
    def handle_bytes_from_mime_type(
        cls,
        content: bytes,
        mime_type: str,
        raise_exception: bool=True
    ):
        if mime_type == 'application/json':
            value = bytes_to_json(content)
        
        else:
            if raise_exception:
                raise Exception(f"Mime Type {mime_type} not allow yet.")
            value = content
        
        return value
        
    def build(self, *args, **kwargs):
        raise NotImplementedError


class DriveFromServiceAcountFile(Drive):
    
    def __init__(self, filename: str, **kwargs) -> None:
        super(DriveFromServiceAcountFile, self).__init__(
            service_account.Credentials.from_service_account_file,
            filename,
            **kwargs
        )
    
    def build(
        self, 
        version: str='v3', 
        *args, 
        **kwargs
    ):
        self.LOGGER.info("Building service...")
        kwargs.pop('credentials', None)
        self.service = build('drive', version, *args, credentials=self.credentials, **kwargs)
        self.LOGGER.info("Building service... Done!")