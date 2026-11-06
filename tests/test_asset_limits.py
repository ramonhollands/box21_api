"""Photo quotas for a train-config dataset download."""

from box21_api.asset_limits import apply_train_asset_limits, limit_annotations_per_label


A, B, C = 10, 20, 30


def _ann(ann_id, label_id, asset_id, name=None, validation=False):
    return {
        "id": ann_id,
        "label_id": label_id,
        "label_name": name or str(label_id),
        "asset_id": asset_id,
        "in_validation_set": validation,
    }


def _assets(anns, label_id):
    return {a["asset_id"] for a in anns if a["label_id"] == label_id}


def test_unset_maximum_keeps_every_photo():
    anns = [_ann(i, A, i) for i in range(1, 6)]
    assert limit_annotations_per_label(anns, max_per_label=0, min_per_label=3) == anns
    assert limit_annotations_per_label(anns, max_per_label=None) == anns
    assert limit_annotations_per_label(anns, max_per_label="") == anns


def test_multiple_boxes_on_one_photo_count_as_one():
    anns = [
        _ann(1, A, 1),
        _ann(2, A, 1),
        _ann(3, A, 1),
        _ann(4, A, 2),
        _ann(5, A, 3),
    ]
    limited = limit_annotations_per_label(anns, max_per_label=1)
    assert {a["asset_id"] for a in limited} == {1}
    assert [a["id"] for a in limited] == [1, 2, 3]


def test_hard_maximum_does_not_overshoot_cooccurring_label():
    anns = []
    for asset_id in range(1, 6):
        anns.append(_ann(asset_id, A, asset_id, "A"))
    for asset_id in range(101, 104):
        anns.append(_ann(100 + asset_id, B, asset_id, "B"))
        anns.append(_ann(200 + asset_id, A, asset_id, "A"))

    limited = limit_annotations_per_label(anns, max_per_label=3)
    assert _assets(limited, A) == {101, 102, 103}
    assert _assets(limited, B) == {101, 102, 103}


def test_minimum_pulls_extra_c_photos_past_maximum():
    """A and B never appear without C. min = max = 4, so C reaches 8."""
    anns = []
    ann_id = 1
    for asset_id in range(1, 5):
        anns.append(_ann(ann_id, C, asset_id, "C"))
        ann_id += 1
    for asset_id in range(5, 9):
        anns.append(_ann(ann_id, A, asset_id, "A"))
        ann_id += 1
        anns.append(_ann(ann_id, C, asset_id, "C"))
        ann_id += 1
    for asset_id in range(9, 13):
        anns.append(_ann(ann_id, B, asset_id, "B"))
        ann_id += 1
        anns.append(_ann(ann_id, C, asset_id, "C"))
        ann_id += 1

    limited = limit_annotations_per_label(anns, max_per_label=4, min_per_label=4)
    assert _assets(limited, A) == {5, 6, 7, 8}
    assert _assets(limited, B) == {9, 10, 11, 12}
    assert _assets(limited, C) == {5, 6, 7, 8, 9, 10, 11, 12}
    assert 1 not in {a["asset_id"] for a in limited}


def test_minimum_stops_before_pushing_c_further_over_its_maximum():
    """Once A and B are at the minimum, the next photo would push C past its maximum."""
    anns = []
    ann_id = 1
    for asset_id in range(1, 5):
        anns.append(_ann(ann_id, C, asset_id, "C"))
        ann_id += 1
    for asset_id in range(5, 9):
        anns.append(_ann(ann_id, A, asset_id, "A"))
        ann_id += 1
        anns.append(_ann(ann_id, C, asset_id, "C"))
        ann_id += 1
    for asset_id in range(9, 13):
        anns.append(_ann(ann_id, B, asset_id, "B"))
        ann_id += 1
        anns.append(_ann(ann_id, C, asset_id, "C"))
        ann_id += 1

    limited = limit_annotations_per_label(anns, max_per_label=4, min_per_label=2)
    assert _assets(limited, A) == {5, 6}
    assert _assets(limited, B) == {9, 10}
    assert _assets(limited, C) == {5, 6, 9, 10}


def test_validation_photos_are_not_limited():
    anns = [
        _ann(1, A, 1),
        _ann(2, A, 2),
        _ann(3, A, 3),
        _ann(4, A, 9, validation=True),
        _ann(5, A, 10, validation=True),
    ]
    kept = apply_train_asset_limits(anns, max_per_label=1, min_per_label=1)
    assert [a["id"] for a in kept] == [1, 4, 5]
