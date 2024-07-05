

from drtools.types import JSONLike
import json


def mime_type_is_folder(mime_type: str) -> bool:
    return 'application/vnd.google-apps.folder' == mime_type

def bytes_to_json(bytes_value: bytes, encoding: str='utf-8') -> JSONLike:
    return json.loads(bytes_value.decode(encoding))