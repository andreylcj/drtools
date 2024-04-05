

from typing import List, TypedDict, Optional


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