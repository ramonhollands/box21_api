"""Photo quotas per label, matching box21_ai_worker limit_annotations_per_label.

A photo counts once for each label on it. Every annotation on a selected
photo is kept. Labels are train labels: a child annotation (Rat under train
label Animal) counts toward Animal only, and Animal's photos are spread evenly
over its child labels (``source_label_id``). The minimum pass may push a co-occurring label over its
maximum. The maximum pass does not. Validation photos are not limited.
"""

import heapq
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union


def _parse_positive_int(value: Optional[Union[int, str, float]]) -> int:
    """Return a positive int, or 0 meaning the bound is unset."""
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _is_validation(annotation: Dict[str, Any]) -> bool:
    return bool(annotation.get("in_validation_set"))


def _source_label_id(annotation: Dict[str, Any]) -> int:
    """Label as annotated (e.g. Rat), or the train label when it is not a child."""
    source = annotation.get("source_label_id")
    if source is None:
        return int(annotation["label_id"])
    return int(source)


def _asset_labels(annotations: List[Dict[str, Any]]) -> Tuple[Dict[int, set], Dict[int, set]]:
    """Map asset id to its train label ids, and to its (train, source) label pairs."""
    labels_by_asset: Dict[int, set] = {}
    groups_by_asset: Dict[int, set] = {}
    for ann in annotations:
        asset_id = ann.get("asset_id")
        label_id = ann.get("label_id")
        if asset_id is None or label_id is None:
            continue
        labels_by_asset.setdefault(int(asset_id), set()).add(int(label_id))
        groups_by_asset.setdefault(int(asset_id), set()).add(
            (int(label_id), _source_label_id(ann))
        )
    return labels_by_asset, groups_by_asset


def _select_assets_toward(
    asset_labels: Dict[int, set],
    asset_groups: Dict[int, set],
    counts: Dict[int, int],
    group_counts: Dict[tuple, int],
    selected: set,
    threshold: int,
    *,
    hard_cap: bool,
) -> None:
    """Greedily add photos until every label on a candidate is blocked.

    Priority, best first:
    1. How many train labels on the photo are still under ``threshold``.
    2. Fewest photos picked so far for the photo's child labels, so a train
       label such as Animal fills evenly with Rat and Vulpes photos.
    3. Lowest asset id.

    ``hard_cap`` is the maximum pass: a photo is skipped when any label on
    it is already at the threshold. The minimum pass leaves this off, so a
    co-occurring label may be pushed over its maximum.
    """
    if threshold <= 0:
        return

    def priority(asset_id: int) -> Optional[Tuple[int, int]]:
        labels = asset_labels[asset_id]
        if hard_cap and any(counts[label_id] >= threshold for label_id in labels):
            return None
        open_labels = {label_id for label_id in labels if counts[label_id] < threshold}
        if not open_labels:
            return None
        balance = min(
            group_counts[group]
            for group in asset_groups[asset_id]
            if group[0] in open_labels
        )
        return (-len(open_labels), balance)

    # Priorities only get worse as photos are picked, so a stale heap entry
    # is pushed back with its new priority when it comes up.
    heap: List[tuple] = []
    for asset_id in asset_labels:
        current = priority(asset_id)
        if current is not None:
            heapq.heappush(heap, (*current, asset_id))

    while heap:
        neg_open, balance, asset_id = heapq.heappop(heap)
        if asset_id in selected:
            continue
        current = priority(asset_id)
        if current is None:
            continue
        if current != (neg_open, balance):
            heapq.heappush(heap, (*current, asset_id))
            continue
        selected.add(asset_id)
        for label_id in asset_labels[asset_id]:
            counts[label_id] += 1
        for group in asset_groups[asset_id]:
            group_counts[group] += 1


def limit_annotations_per_label(
    annotations: List[Dict[str, Any]],
    max_per_label: Optional[Union[int, str, float]] = None,
    min_per_label: Optional[Union[int, str, float]] = None,
) -> List[Dict[str, Any]]:
    """Keep a bounded number of photos per label. See module docstring."""
    max_n = _parse_positive_int(max_per_label)
    if max_n <= 0 or not annotations:
        return list(annotations)

    min_n = _parse_positive_int(min_per_label)
    asset_labels, asset_groups = _asset_labels(annotations)
    counts: Dict[int, int] = defaultdict(int)
    group_counts: Dict[tuple, int] = defaultdict(int)
    selected: set = set()

    if min_n > 0:
        _select_assets_toward(
            asset_labels, asset_groups, counts, group_counts, selected,
            min_n, hard_cap=False,
        )
    _select_assets_toward(
        asset_labels, asset_groups, counts, group_counts, selected,
        max_n, hard_cap=True,
    )

    if not selected:
        return []

    return [
        ann
        for ann in annotations
        if ann.get("asset_id") is not None and int(ann["asset_id"]) in selected
    ]


def log_assets_per_label(
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
    max_per_label: Optional[Union[int, str, float]] = None,
    min_per_label: Optional[Union[int, str, float]] = None,
) -> None:
    """Log how many training photos each label kept, including shortfalls."""
    max_n = _parse_positive_int(max_per_label)
    min_n = _parse_positive_int(min_per_label)
    if max_n <= 0 and min_n <= 0:
        return

    names: Dict[int, str] = {}
    before_assets: Dict[int, set] = defaultdict(set)
    for ann in before:
        label_id = ann.get("label_id")
        asset_id = ann.get("asset_id")
        if label_id is None or asset_id is None:
            continue
        before_assets[int(label_id)].add(int(asset_id))
        name = ann.get("label_name")
        if name and int(label_id) not in names:
            names[int(label_id)] = str(name)

    after_assets: Dict[int, set] = defaultdict(set)
    after_children: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    child_names: Dict[int, str] = {}
    for ann in after:
        label_id = ann.get("label_id")
        asset_id = ann.get("asset_id")
        if label_id is None or asset_id is None:
            continue
        after_assets[int(label_id)].add(int(asset_id))
        source_id = _source_label_id(ann)
        if source_id != int(label_id):
            after_children[int(label_id)][source_id].add(int(asset_id))
            if ann.get("source_label_name"):
                child_names[source_id] = str(ann["source_label_name"])

    max_label = "none" if max_n <= 0 else str(max_n)
    min_label = "none" if min_n <= 0 else str(min_n)
    print(
        f"Applied assets per label (min={min_label}, max={max_label}): "
        f"train annotations {len(before)} -> {len(after)}",
        flush=True,
    )
    for label_id in sorted(before_assets):
        kept = len(after_assets.get(label_id, ()))
        name = names.get(label_id, str(label_id))
        shortfall = ""
        if min_n > 0 and kept < min_n:
            shortfall = f", below minimum {min_n}"
        print(
            f"  {name} (id={label_id}): {kept} photos{shortfall}",
            flush=True,
        )
        for child_id in sorted(after_children.get(label_id, {})):
            child_kept = len(after_children[label_id][child_id])
            child_name = child_names.get(child_id, str(child_id))
            print(f"    {child_name} (id={child_id}): {child_kept} photos", flush=True)


def bounds_from_meta(meta: Optional[Dict[str, Any]]) -> tuple:
    """Return ``(max_labels, min_labels)`` raw values from a train-config meta dict."""
    meta = meta or {}
    return meta.get("max_labels"), meta.get("min_labels")


def apply_train_asset_limits(
    annotations: List[Dict[str, Any]],
    max_per_label: Optional[Union[int, str, float]] = None,
    min_per_label: Optional[Union[int, str, float]] = None,
) -> List[Dict[str, Any]]:
    """Limit training photos. Validation annotations stay in their original place."""
    train = [ann for ann in annotations if not _is_validation(ann)]
    limited_train = limit_annotations_per_label(
        train,
        max_per_label=max_per_label,
        min_per_label=min_per_label,
    )
    log_assets_per_label(
        train,
        limited_train,
        max_per_label=max_per_label,
        min_per_label=min_per_label,
    )
    if _parse_positive_int(max_per_label) <= 0:
        return list(annotations)

    selected = {
        int(ann["asset_id"])
        for ann in limited_train
        if ann.get("asset_id") is not None
    }
    return [
        ann
        for ann in annotations
        if _is_validation(ann)
        or (ann.get("asset_id") is not None and int(ann["asset_id"]) in selected)
    ]
