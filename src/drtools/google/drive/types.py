

from typing import List, TypedDict, Optional
from enum import Enum


class Mimetype(Enum):
    """Enum dos tipos MIME suportados pelo módulo Google Drive.

    Cada membro armazena o valor como uma tupla de um elemento, e a property
    :attr:`content_type` retorna a string do tipo MIME para uso direto em
    headers e metadados da API.

    Members:
        JSON: ``application/json``
        CSV: ``text/csv``

    Example:
        >>> Mimetype.JSON.content_type
        'application/json'
        >>> Mimetype.CSV.content_type
        'text/csv'
    """

    JSON = ('application/json',)
    CSV = ('text/csv',)

    @property
    def content_type(self) -> str:
        """Retorna a string do tipo MIME (primeiro elemento da tupla de valor)."""
        return self.value[0]


FileId = str
"""Alias para o ID de arquivo/pasta do Google Drive.

IDs são strings opacas retornadas pela API do Drive (ex: ``"1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"``).

Example:
    >>> file_id: FileId = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
"""


class DefaultFilesListItem(TypedDict):
    """Estrutura de um item retornado pela API de listagem de arquivos do Drive.

    Corresponde aos campos padrão definidos em ``DEFAULTS_FIELDS`` em [drive.py](drive.py).

    Attributes:
        name: Nome do arquivo ou pasta.
        id: ID único do arquivo no Google Drive.
        kind: Tipo de recurso (tipicamente ``"drive#file"``).
        mimeType: Tipo MIME do arquivo (e.g. ``"text/csv"``, ``"application/vnd.google-apps.folder"``).
        createdTime: Data/hora de criação no formato RFC 3339 (e.g. ``"2026-01-15T10:30:00.000Z"``).
        modifiedTime: Data/hora da última modificação no formato RFC 3339.
    """

    name: str
    id: FileId
    kind: str
    mimeType: str
    createdTime: str
    modifiedTime: str


class DefaultFilesListResult(TypedDict):
    """Estrutura do resultado paginado da API de listagem de arquivos do Drive.

    Attributes:
        files: Lista de itens encontrados na página atual.
        nextPageToken: Token para buscar a próxima página. ``None`` se for a última página.

    Example:
        >>> result: DefaultFilesListResult = {
        ...     "files": [
        ...         {
        ...             "name": "relatorio.csv",
        ...             "id": "abc123",
        ...             "kind": "drive#file",
        ...             "mimeType": "text/csv",
        ...             "createdTime": "2026-01-10T08:00:00.000Z",
        ...             "modifiedTime": "2026-03-19T14:22:00.000Z",
        ...         }
        ...     ],
        ...     "nextPageToken": None,
        ... }
    """

    files: List[DefaultFilesListItem]
    nextPageToken: Optional[str]


class FilesListItem(DefaultFilesListItem):
    """Item de arquivo/pasta retornado pelos métodos de listagem do :class:`~drtools.google.drive.drive.Drive`.

    Extensão de :class:`DefaultFilesListItem` reservada para adição de campos
    customizados futuramente.
    """
    pass


class FilesListResult(DefaultFilesListResult):
    """Resultado paginado retornado pelos métodos de listagem do :class:`~drtools.google.drive.drive.Drive`.

    Extensão de :class:`DefaultFilesListResult` reservada para adição de campos
    customizados futuramente.
    """
    pass
