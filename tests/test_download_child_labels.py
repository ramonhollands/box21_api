"""Train-config download maps child annotations to their train label, like the worker."""

from types import SimpleNamespace

from box21_api import box21_download_dataset as dl


ANIMAL, RAT, VULPES = 500, 501, 502


def _api_ann(ann_id, label_id, label_name, parent_id, asset_id):
    return {
        "id": ann_id,
        "coords": "[0.1, 0.1, 0.2, 0.2]",
        "relationships": {
            "asset": {"id": asset_id, "in_validation_set": False, "filename": f"{asset_id}.jpg"},
            "label": {"id": label_id, "name": label_name, "parent_id": parent_id},
        },
    }


# Boxes in the project: Animal itself, a Rat and a Vulpes (children of Animal).
BOXES = [
    _api_ann(1, ANIMAL, "Animal", None, 10),
    _api_ann(2, RAT, "Rat", ANIMAL, 11),
    _api_ann(3, VULPES, "Vulpes", ANIMAL, 12),
]


class FakeApi:
    def __init__(self, label_set_ids):
        self.label_set_ids = label_set_ids

    def get_label_set(self, label_set_id, project_id):
        return {"relationships": {"label_set_labels": [{"label_id": i} for i in self.label_set_ids]}}

    def get_labels(self, label_ids):
        names = {ANIMAL: ("Animal", None), RAT: ("Rat", ANIMAL), VULPES: ("Vulpes", ANIMAL)}
        return [SimpleNamespace(id=i, name=names[i][0], parent_id=names[i][1]) for i in label_ids]

    def filter_boxes(self, filters, offset, limit, project_id):
        label = next(f["value"] for f in filters if f["type"] == 27)
        validation = next(f["value"] for f in filters if f["type"] == 13) == "Yes"
        if validation:
            return {"annotations": []}
        # Filter 27: the label and its descendants.
        if label == "Animal":
            return {"annotations": BOXES}
        return {"annotations": [b for b in BOXES if b["relationships"]["label"]["name"] == label]}


def _config():
    return {"name": "cfg", "meta": '{"label_set_id": 1}'}


def test_children_are_mapped_to_train_label():
    anns = dl.get_annotations_for_train_config(FakeApi([ANIMAL]), 1, _config())
    assert [(a["label_id"], a["label_name"]) for a in anns] == [(ANIMAL, "Animal")] * 3
    assert [a["source_label_id"] for a in anns] == [None, RAT, VULPES]
    assert [a["source_label_name"] for a in anns] == [None, "Rat", "Vulpes"]


def test_child_in_label_set_keeps_its_own_label_once():
    anns = dl.get_annotations_for_train_config(FakeApi([ANIMAL, RAT]), 1, _config())
    by_ann = {a["annotation_id"]: a for a in anns}
    assert len(anns) == 3
    assert by_ann[2]["label_name"] == "Rat" and by_ann[2]["source_label_id"] is None
    assert by_ann[3]["label_name"] == "Animal" and by_ann[3]["source_label_id"] == VULPES


def test_overlapping_filter_sets_still_keep_same_label_copies():
    anns = [
        {"annotation_id": 1, "label_id": ANIMAL, "source_label_id": None, "label_parent_id": None},
        {"annotation_id": 1, "label_id": ANIMAL, "source_label_id": None, "label_parent_id": None},
    ]
    assert len(dl._keep_nearest_train_label(anns)) == 2
