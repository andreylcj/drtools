

from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import List, Tuple, Dict, Union
from .types import (
    FileId,
    FilesListResult,
    FilesListItem,
    Mimetype,
)
from .utils import (
    bytes_to_json
)
import io
from googleapiclient.http import (
    MediaIoBaseDownload, 
    MediaFileUpload, 
    DEFAULT_CHUNK_SIZE,
    MediaIoBaseUpload,
)
from drtools.logging import Logger, FormatterOptions
from typing import Callable
import json


DEFAULTS_FIELDS: str = "nextPageToken, files(id, name, kind, mimeType, createdTime, modifiedTime)"
DEFAULT_ORDER_BY: str = "modifiedTime desc"


class Drive:
    
    SCOPES: List[str] = ['https://www.googleapis.com/auth/drive']
    
    def __init__(
        self,
        credentials_method: Callable,
        *args,
        LOGGER: Logger=None,
        **kwargs,
    ) -> None:
        kwargs['scopes'] = kwargs.get('scopes', self.SCOPES)
        self.credentials = credentials_method(*args, **kwargs)
        if not LOGGER:
            LOGGER = Logger(
                name="Drive",
                formatter_options=FormatterOptions(include_datetime=True, include_logger_name=True, include_level_name=True),
                default_start=False
            )
        self.LOGGER = LOGGER
        self._service = None
        
    def set_service(self, service) -> None:
        self._service = service
    
    @property
    def service(self):
        return self._service
        
    def get_folders_from_name(
        self,
        name: str, 
        page_size: int=10,
        fields: str=DEFAULTS_FIELDS,
        parent_folder_id: str=None,
        order_by: str=DEFAULT_ORDER_BY
    ) -> FilesListResult:
        """List folders and files in Google Drive."""
        q = f"mimeType='application/vnd.google-apps.folder' and name='{name}'"
        if parent_folder_id:
            q += f" and '{parent_folder_id}' in parents"
        results = self.service.files().list(
            q=q, 
            pageSize=page_size, 
            fields=fields,
            orderBy=order_by,
        ).execute()
        return results
    
    def get_folder_content(
        self,
        folder_id, 
        page_size: int=1000, 
        trashed: bool=False, 
        fields: str=DEFAULTS_FIELDS,
        deep: bool=False,
        order_by: str=DEFAULT_ORDER_BY,
    ) -> FilesListResult:
        """List folders and files in Google Drive."""
        def _get_folder_content(folder_id, page_size, trashed):
            trashed = 'true' if trashed else 'false'
            results = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed={trashed}", 
                pageSize=page_size, 
                fields=fields,
                orderBy=order_by,
            ).execute()
            items = results
            return items
        items = _get_folder_content(folder_id, page_size, trashed)
        if deep:
            items = [
                {**item, 'content': self.serive.get_folder_content(item['id'], page_size, trashed, True) if 'folder' in item['mimeType'] else None}
                for item in items
            ]
        return items
    
    def get_folder_id_from_path(self, path: str) -> FileId:
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
        return parent_id

    def get_folder_content_from_path(
        self,
        path: str, 
        page_size: int=1000, 
        trashed: bool=False, 
        fields: str=DEFAULTS_FIELDS,
        deep: bool=False,
        order_by: str=DEFAULT_ORDER_BY,
    ) -> FilesListResult:
        """List folders and files in Google Drive."""
        folder_id = self.get_folder_id_from_path(path)
        return self.get_folder_content(folder_id, page_size, trashed, fields, deep, order_by)
    
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
            if self.children_exists(folder_name, parent_folder_id):
                return None
        created_folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        return created_folder["id"]
    
    def create_folder_from_path(
        self,
        folder_path: str, 
        ignore_if_exists: bool=True
    ) -> FileId:
        parent, name = self.get_parent_and_name_from_path(folder_path)
        parent_id = self.get_folder_id_from_path(parent)
        return self.create_folder(name, parent_id, ignore_if_exists)

    def get_file_from_path(self, path: str) -> FilesListItem:
        folder_path, filename = self.get_parent_and_name_from_path(path)
        folder_content = self.get_folder_content_from_path(folder_path)
        file_item = [item for item in folder_content['files'] if item['name'] == filename]
        if len(file_item) > 1:
            raise Exception(f"There are more than 1 file with same path. Files founde: {len(file_item):,}")
        if len(file_item) == 0:
            raise Exception(f"No file found with path: {path}")
        return file_item[0]

    def get_file_id_from_path(self, path: str) -> FileId:
        return self.get_file_from_path(path)['id']
    
    def get_file_content(
        self,
        file_id: str,
        try_handle_mimetype: bool=True,
        mimetype: str=None,
    ) -> bytes:
        request = self.service.files().get_media(fileId=file_id)
        # create_directories_of_path(filepath)
        fh = io.BytesIO()
        # fh = io.FileIO(filepath, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            self.LOGGER.debug(f"[GoogleDrive:FileID:{file_id}] Download {int(status.progress() * 100)}%.")
        value = fh.getvalue()
        if try_handle_mimetype:
            value = self.handle_bytes_from_mimetype(value, mimetype)
        return value
    
    @classmethod
    def get_parent_and_name_from_path(cls, filepath: str) -> Tuple[str, str]:
        parent = '/'.join(filepath.split('/')[:-1])
        name = filepath.split('/')[-1]
        return parent, name
    
    def children_exists(self, name: str, parent_folder_id: str) -> bool:
        content = self.get_folder_content(parent_folder_id)
        file_item = [item for item in content['files'] if item['name'] == name]
        return len(file_item) > 0
    
    def path_exists(self, path: str) -> bool:
        parent, name = self.get_parent_and_name_from_path(path)
        content = self.get_folder_content_from_path(parent)
        file_item = [item for item in content['files'] if item['name'] == name]
        return len(file_item) > 0
    
    def get_file_content_from_path(
        self, 
        filepath: str,
        try_handle_mimetype: bool=True,
    ) -> io.BytesIO:
        folder_path, filename = self.get_parent_and_name_from_path(filepath)
        self_folder_content = self.get_folder_content_from_path(folder_path)
        file_item = [item for item in self_folder_content['files'] if item['name'] == filename]
        if len(file_item) > 1:
            raise Exception(f"There are more than 1 file with same path. Files founde: {len(file_item):,}")
        if len(file_item) == 0:
            raise Exception(f"No file found with path: {filepath}")
        file_id = file_item[0]['id']
        return self.get_file_content(file_id, try_handle_mimetype, file_item[0]['mimeType'])
    
    def get_last_modified_file_content_from_folder(self, folder_path: str, mimetype: str=None):
        folder_content = self.get_folder_content_from_path(folder_path, page_size=1)
        file_id = folder_content['files'][0]['id']
        mimetype = folder_content['files'][0]['mimeType']
        data = self.get_file_content(file_id, mimetype=mimetype)
        return data
    
    def create_file(
        self,
        name: str, 
        parent_folder_id: FileId, 
        media: Union[MediaFileUpload, MediaIoBaseUpload]=None,
        ignore_if_exists: bool=True
    ) -> FileId:
        """Create a file in Google Drive and return its ID."""
        file_metadata = {
            'name': name,
            'parents': [parent_folder_id] if parent_folder_id else []
        }
        if ignore_if_exists:
            if self.children_exists(name, parent_folder_id):
                self.LOGGER.debug(f'File with name {name} inside folder with id {parent_folder_id} already exists')
                return None
        kwargs = {'body': file_metadata, 'fields': 'id'}
        if media:
            kwargs['media_body'] = media
        created_file = self.service.files().create(**kwargs).execute()
        return created_file["id"]
    
    def create_file_from_media_file(
        self,
        filepath: str, 
        filename: str,
        mimetype: str=None,
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
        ignore_if_exists: bool=True
    ) -> FileId:
        parent, name = self.get_parent_and_name_from_path(filepath)
        parent_id = self.get_folder_id_from_path(parent)
        media = MediaFileUpload(filename, mimetype, chunksize, resumable)
        return self.create_file(name, parent_id, media, ignore_if_exists)
    
    def create_file_from_media_io(
        self,
        content_bytes: bytes, 
        filepath: str,
        mimetype: str,
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
        ignore_if_exists: bool=True
    ) -> FileId:
        parent, name = self.get_parent_and_name_from_path(filepath)
        parent_id = self.get_folder_id_from_path(parent)
        content_stream = io.BytesIO(content_bytes)
        media = MediaIoBaseUpload(content_stream, mimetype, chunksize, resumable)
        return self.create_file(name, parent_id, media, ignore_if_exists)
    
    def upload_dict(
        self, 
        data: Dict, 
        filepath: str,
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
        ignore_if_exists: bool=True,
    ):
        content_bytes = json.dumps(data).encode('utf-8')
        return self.create_file_from_media_io(
            content_bytes, 
            filepath, 
            Mimetype.JSON.content_type, 
            chunksize, 
            resumable, 
            ignore_if_exists
        )
    
    @classmethod
    def handle_bytes_from_mimetype(
        cls,
        content: bytes,
        mimetype: str,
        raise_exception: bool=True
    ):
        if mimetype == Mimetype.JSON.content_type:
            value = bytes_to_json(content)
        
        else:
            if raise_exception:
                raise Exception(f"Mime Type {mimetype} not allow yet.")
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
        kwargs['credentials'] = self.credentials
        self.set_service(build('drive', version, *args, **kwargs))
        self.LOGGER.info("Building service... Done!")