

from drtools.types import JSONLike, DictKey, DictValue
import json
from typing import List, Dict, Optional
import csv


def mime_type_is_folder(mime_type: str) -> bool:
    return 'application/vnd.google-apps.folder' == mime_type

def bytes_to_json(bytes_value: bytes, encoding: str='utf-8') -> JSONLike:
    return json.loads(bytes_value.decode(encoding))

def bytes_to_csv_dicts(
    bytes_value: bytes, 
    encoding: str = 'utf-8',
    delimiter: str = ',',
    header: Optional[List[str]] = None,
    skiprows: int = 0
) -> List[Dict[DictKey, DictValue]]:
    """
    Converte bytes de CSV para uma lista de dicionários.

    Args:
        bytes_value (bytes): Conteúdo do CSV em bytes.
        encoding (str): Codificação do conteúdo (padrão: 'utf-8').
        delimiter (str): Separador de campos (padrão: ',').
        header (List[str], opcional): Cabeçalho personalizado, se o CSV não tiver.
        skiprows (int): Número de linhas iniciais a pular antes do cabeçalho (padrão: 0).

    Returns:
        List[Dict[str, str]]: Lista de dicionários com os dados do CSV.
    """
    text = bytes_value.decode(encoding)
    lines = text.splitlines()
    # Pula as linhas solicitadas
    lines = lines[skiprows:]
    # Usa o cabeçalho fornecido ou pega da primeira linha
    if header is not None:
        reader = csv.DictReader(lines, fieldnames=header, delimiter=delimiter)
    else:
        reader = csv.DictReader(lines, delimiter=delimiter)
    return list(reader)