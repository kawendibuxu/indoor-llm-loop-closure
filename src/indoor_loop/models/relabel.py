ANCHOR_CLASS_MAP = {
    "bed": "bed",
    "desk": "table_like",
    "table": "table_like",
    "chair": "seat_like",
    "sofa": "seat_like",
    "cabinet": "storage_like",
    "bookshelf": "storage_like",
    "shelf": "storage_like",
    "monitor": "display_like",
    "tv": "display_like",
    "sink": "sanitary_like",
    "toilet": "sanitary_like",
    "window": "opening_like",
    "door": "opening_like",
}


def map_to_anchor_class(label: str) -> str:
    return ANCHOR_CLASS_MAP.get(label, "other")
