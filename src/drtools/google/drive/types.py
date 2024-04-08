

from typing import List, TypedDict, Optional
from enum import Enum


class Mimetype(Enum):
    JSON = ('application/json',)
    
    @property
    def content_type(self) -> str:
        return self.value[0]


FileId = str


class DefaultFilesListItem(TypedDict):
    name: str
    id: FileId
    kind: str
    mimeType: str

class DefaultFilesListResult(TypedDict):
    files: List[DefaultFilesListItem]
    nextPageToken: Optional[str]

class FilesListItem(DefaultFilesListItem):
    pass

class FilesListResult(DefaultFilesListResult):
    pass