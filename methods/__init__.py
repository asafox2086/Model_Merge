from .lamp_merge import merge_lamp_merge


METHOD_ALIASES = {
    "lamp_merge": "lamp_merge",
    "lamp-merge": "lamp_merge",
    "lampmerge": "lamp_merge",
}


def normalize_method_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(
            f"Unsupported merge method: {name}. "
            "This public release contains only the LAMP-Merge pipeline."
        )
    return METHOD_ALIASES[key]


__all__ = [
    "merge_lamp_merge",
    "normalize_method_name",
]
