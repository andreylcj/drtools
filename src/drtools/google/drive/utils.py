

from drtools.types import JSONLike, DictKey, DictValue
import json
from typing import List, Dict, Optional
import csv
import io


def mime_type_is_folder(mime_type: str) -> bool:
    """Verifica se um tipo MIME corresponde a uma pasta do Google Drive.

    Args:
        mime_type: String do tipo MIME a verificar.

    Returns:
        ``True`` se ``mime_type`` for ``"application/vnd.google-apps.folder"``,
        ``False`` caso contrário.

    Example:
        >>> mime_type_is_folder("application/vnd.google-apps.folder")
        True
        >>> mime_type_is_folder("text/csv")
        False
    """
    return 'application/vnd.google-apps.folder' == mime_type


def bytes_to_json(bytes_value: bytes, encoding: str='utf-8') -> JSONLike:
    """Decodifica ``bytes`` e faz parse do conteúdo como JSON.

    Args:
        bytes_value: Conteúdo binário de um arquivo JSON.
        encoding: Codificação de caracteres. Padrão: ``"utf-8"``.

    Returns:
        Objeto Python resultante do parse JSON (dict, list, etc.).

    Example:
        >>> data = bytes_to_json(b'{"key": "value"}')
        >>> data
        {'key': 'value'}

        >>> data = bytes_to_json(b'[1, 2, 3]')
        >>> data
        [1, 2, 3]
    """
    return json.loads(bytes_value.decode(encoding))


def bytes_to_csv_dicts(
    bytes_value: bytes,
    encoding: str = 'utf-8',
    delimiter: str = ',',
    header: Optional[List[str]] = None,
    skiprows: int = 0
) -> List[Dict[DictKey, DictValue]]:
    """Converte bytes de CSV para uma lista de dicionários.

    Decodifica o conteúdo binário, divide em linhas, pula as linhas iniciais
    desejadas e usa :class:`csv.DictReader` para construir a lista de registros.

    Args:
        bytes_value: Conteúdo do arquivo CSV em bytes.
        encoding: Codificação do conteúdo. Padrão: ``"utf-8"``.
        delimiter: Separador de campos. Padrão: ``","`` (vírgula).
        header: Cabeçalho personalizado (lista de nomes de colunas).
            Se ``None``, usa a primeira linha do CSV como cabeçalho.
        skiprows: Número de linhas a pular **antes** do cabeçalho. Padrão: ``0``.

    Returns:
        Lista de dicionários, onde cada dict representa uma linha do CSV com
        as chaves sendo os nomes das colunas.

    Example:
        >>> csv_bytes = b"nome,idade\\nAlice,30\\nBob,25"
        >>> records = bytes_to_csv_dicts(csv_bytes)
        >>> records
        [{'nome': 'Alice', 'idade': '30'}, {'nome': 'Bob', 'idade': '25'}]

        >>> # CSV com separador ponto-e-vírgula
        >>> csv_bytes = b"nome;cidade\\nAlice;SP\\nBob;RJ"
        >>> records = bytes_to_csv_dicts(csv_bytes, delimiter=";")
        >>> records
        [{'nome': 'Alice', 'cidade': 'SP'}, {'nome': 'Bob', 'cidade': 'RJ'}]

        >>> # CSV sem cabeçalho, fornecer manualmente
        >>> csv_bytes = b"Alice,30\\nBob,25"
        >>> records = bytes_to_csv_dicts(csv_bytes, header=["nome", "idade"])
        >>> records
        [{'nome': 'Alice', 'idade': '30'}, {'nome': 'Bob', 'idade': '25'}]

        >>> # Pular linhas de metadados no início do arquivo
        >>> csv_bytes = b"# gerado em 2026-03-20\\nnome,idade\\nAlice,30"
        >>> records = bytes_to_csv_dicts(csv_bytes, skiprows=1)
        >>> records
        [{'nome': 'Alice', 'idade': '30'}]
    """
    csv_string = bytes_value.decode(encoding)
    file_in_memory = io.StringIO(csv_string)
    if skiprows > 0:
        for _ in range(skiprows):
            next(file_in_memory, None)
    kwargs = {}
    if header:
        kwargs['fieldnames'] = header
    dict_reader = csv.DictReader(file_in_memory, delimiter=delimiter, quotechar='"', **kwargs)
    records = list(dict_reader)
    return records
    
    # lines = csv_string.splitlines()
    # # Pula as linhas solicitadas
    # lines = lines[skiprows:]
    # # Usa o cabeçalho fornecido ou pega da primeira linha
    # if header is not None:
    #     reader = csv.DictReader(lines, fieldnames=header, delimiter=delimiter)
    # else:
    #     reader = csv.DictReader(lines, delimiter=delimiter)
    # return list(reader)
