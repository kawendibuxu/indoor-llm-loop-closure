from indoor_loop.models.relabel import map_to_anchor_class


def test_map_to_anchor_class_merges_fine_labels() -> None:
    assert map_to_anchor_class("desk") == "table_like"
    assert map_to_anchor_class("table") == "table_like"
    assert map_to_anchor_class("bookshelf") == "storage_like"
