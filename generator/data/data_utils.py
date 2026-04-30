import gzip
import json
import pickle
from typing import Any, Dict, List, Union

def save_compressed_pickle(obj: Any, filepath: str) -> None:
    """
    Save a Python object to a compressed .pkl.gz file.
    """
    with gzip.open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_compressed_pickle(filepath: str) -> Any:
    """
    Load a Python object from a compressed .pkl.gz file.
    """
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)

# ===== NEW: JSON helpers =====

def save_compressed_json(
    obj: Any,
    filepath: str,
    *,
    indent: int | None = None,
    ensure_ascii: bool = False
) -> None:
    """
    Save a JSON-serializable object to a compressed .json.gz file.

    Notes
    -----
    - JSON will serialize dict keys as strings. If your data uses integer keys
      (e.g., {window: {partition: [...]}}), consider setting
      `coerce_int_keys=True` when loading to restore ints.
    """
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)

def _coerce_json_numeric_keys_to_ints(x: Any) -> Any:
    """
    Recursively convert dict keys that look like integers back to int.
    Leaves non-numeric keys unchanged.
    """
    if isinstance(x, dict):
        out: Dict[Any, Any] = {}
        for k, v in x.items():
            if isinstance(k, str) and k.isdigit():
                kk: Union[str, int] = int(k)
            else:
                kk = k
            out[kk] = _coerce_json_numeric_keys_to_ints(v)
        return out
    elif isinstance(x, list):
        return [_coerce_json_numeric_keys_to_ints(v) for v in x]
    else:
        return x

def load_compressed_json(
    filepath: str,
    *,
    coerce_int_keys: bool = False
) -> Any:
    """
    Load a JSON object from a compressed .json.gz file.

    Parameters
    ----------
    coerce_int_keys : bool
        If True, recursively convert dict keys that are decimal strings
        (e.g., "0", "1") back to integers. Handy for data shaped like
        {window: {partition: [ordered_keys]}}.
    """
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    if coerce_int_keys:
        obj = _coerce_json_numeric_keys_to_ints(obj)
    return obj
