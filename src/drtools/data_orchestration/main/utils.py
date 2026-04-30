

from collections import Counter
from typing import List, Tuple


def has_unique_elements(arr: list) -> bool:
    """Return True if all elements in the list are unique."""
    return len(arr) == len(set(arr))


def get_duplicates(arr: list[str]) -> list[str]:
    """Return a list of elements that appear more than once in arr."""
    counts = Counter(arr)
    return [item for item, freq in counts.items() if freq > 1]


def generate_pipeline_names(
    src_list: List[Tuple[str]],
    dst_list: List[Tuple[str]],
    data_type_src=None,
    data_type_dst=None
):
    """Generate a file name and class name for an ETL pipeline.

    Each item in src_list and dst_list must be a tuple of 2 or 3 elements:
        (sigla, folder_name) or (sigla, folder_name, version)

    Args:
        src_list: List of source descriptor tuples.
        dst_list: List of destination descriptor tuples.
        data_type_src: Optional data type label for the source.
        data_type_dst: Optional data type label for the destination.

    Returns:
        Tuple (file_name, class_name), e.g.:
            ('gd_ads_go_perf_v1_to_gs_perf_ads.py', 'GdAdsGoPerfV1ToGsPerfAds')

    Raises:
        Exception: If any tuple has fewer than 2 or more than 3 elements.
    """

    # --- Construindo a parte da origem ---
    src_parts = []
    src_cls_parts = []
    for item in src_list:
        sigla, pasta, versao = None, None, None
        if len(item) > 3:
            raise Exception("Too many values. Expected 3 at maximum")
        elif len(item) == 3:
            sigla, pasta, versao = item
        elif len(item) == 2:
            sigla, pasta = item
        else:
            raise Exception("Too little values. Expected 2 at minimum")
        part = f"{sigla}_{pasta}"
        if versao:
            part += f"_{versao}"
            cls_part = f"{sigla.capitalize()}{pasta.title().replace('_','')}{versao.capitalize()}"
        else:
            cls_part = f"{sigla.capitalize()}{pasta.title().replace('_','')}"
        src_parts.append(part)
        src_cls_parts.append(cls_part)

    # --- Construindo a parte do destino ---
    dst_parts = []
    dst_cls_parts = []
    for item in dst_list:
        sigla, pasta, versao = None, None, None
        if len(item) > 3:
            raise Exception("Too many values. Expected 3 at maximum")
        elif len(item) == 3:
            sigla, pasta, versao = item
        elif len(item) == 2:
            sigla, pasta = item
        else:
            raise Exception("Too little values. Expected 2 at minimum")
        part = f"{sigla}_{pasta}"
        if versao:
            part += f"_{versao}"
            cls_part = f"{sigla.capitalize()}{pasta.title().replace('_','')}{versao.capitalize()}"
        else:
            cls_part = f"{sigla.capitalize()}{pasta.title().replace('_','')}"
        dst_parts.append(part)
        dst_cls_parts.append(cls_part)

    # --- Nome do arquivo ---
    file_name = "_".join(src_parts) + "_to_" + "_".join(dst_parts) + ".py"

    # --- Nome da classe ---
    class_name = "".join(src_cls_parts) + "To" + "".join(dst_cls_parts)

    return file_name, class_name
