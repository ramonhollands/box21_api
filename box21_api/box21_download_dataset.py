"""
Download a dataset for a train config the same way box21_ai_worker would for training.

Given a train_config_id and project_id, resolves labels and filters from the config,
fetches annotations via the filter-boxes API, downloads asset images, and writes
a manifest (labels.txt, train/valid split, summary) so the layout matches what
the worker would use for that config.
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from PIL import Image, ImageDraw
from tqdm import tqdm

from .box21_api import Box21Api
from .box21_export import get_user_input, get_password, select_from_list


# --- Filter building (mirrors box21_ai_worker Box21Dataset.filter_boxes) ---

def _base_filters_for_label(label_name: str, only_valid_assets: bool = True) -> List[Dict[str, Any]]:
    """Base filters for one label: label name, validated, not unclear."""
    if only_valid_assets:
        return [
            {"type": 27, "value": str(label_name)},
            {"type": 9, "value": "Yes"},
            {"type": 11, "value": "No"},
        ]
    return [
        {"type": 27, "value": str(label_name)},
        {"type": 17, "value": "Yes"},
        {"type": 11, "value": "No"},
    ]


def _parse_filter_sets(raw_filters_json: Optional[str]) -> Optional[List[List[Dict[str, Any]]]]:
    """Parse training_set_filters or validation_set_filters from train config meta."""
    if not raw_filters_json or raw_filters_json.strip() == "":
        return None
    try:
        return json.loads(raw_filters_json)
    except json.JSONDecodeError:
        return None


def _build_filters(
    label_name: str,
    in_validation_set: bool,
    additional_filter_sets: Optional[List[List[Dict[str, Any]]]] = None,
    only_valid_assets: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Build filter lists for one label and train/val split.
    Returns a list of filter lists (one per additional_filter_set, or one default).
    """
    base = _base_filters_for_label(label_name, only_valid_assets)
    base.append({"type": 13, "value": "Yes" if in_validation_set else "No"})

    # Worker: when additional_filter_sets is None or empty, use base only
    if not additional_filter_sets:
        return [base]

    result = []
    for extra_list in additional_filter_sets:
        normalized_extra = [{"type": int(f["type"]), "value": str(f["value"])} for f in extra_list]
        override_types = {f["type"] for f in normalized_extra}
        # If train-config filters include a filter type already present in base,
        # prefer the train-config value (avoid impossible AND like type=9 Yes + No).
        base_without_overridden = [f for f in base if int(f["type"]) not in override_types]
        combined = base_without_overridden + normalized_extra
        result.append(combined)
    return result


def _fetch_annotations_for_label(
    api: Box21Api,
    project_id: int,
    label_name: str,
    in_validation_set: bool,
    additional_filter_sets: Optional[List[List[Dict[str, Any]]]] = None,
    only_valid_assets: bool = True,
) -> List[Dict[str, Any]]:
    """Call filter-boxes for one label and train/val, return flat list of annotation records."""
    filter_lists = _build_filters(
        label_name, in_validation_set, additional_filter_sets, only_valid_assets
    )
    out: List[Dict[str, Any]] = []
    limit = 10000
    split_name = "validation" if in_validation_set else "train"
    for idx, filters in enumerate(filter_lists, start=1):
        print(
            f"filter_boxes filters [{split_name}] label='{label_name}' set={idx}/{len(filter_lists)}:"
            f" {json.dumps(filters, ensure_ascii=True)}"
        )
        offset = 0
        while True:
            data = api.filter_boxes(
                filters=filters,
                offset=offset,
                limit=limit,
                project_id=project_id,
            )
            if not data:
                break
            anns = data.get("annotations") or []
            for a in anns:
                rel = a.get("relationships") or {}
                asset_json = rel.get("asset") or {}
                label_json = rel.get("label") or {}
                out.append({
                    "asset_id": asset_json.get("id"),
                    "label_id": label_json.get("id"),
                    "label_name": label_json.get("name"),
                    "label_parent_id": label_json.get("parent_id"),
                    "coords": a.get("coords"),
                    "in_validation_set": asset_json.get("in_validation_set", False),
                    "filename": asset_json.get("filename", ""),
                })
            if len(anns) < limit:
                break
            offset += limit
    return out


def _resolve_train_label_ids(api: Box21Api, project_id: int, train_config: Dict[str, Any]) -> List[int]:
    """Resolve label IDs from train config: label_set or label_ids/selected_label_ids, plus crop_label_id."""
    meta = train_config.get("meta")
    if not meta:
        return []
    try:
        meta_dict = json.loads(meta) if isinstance(meta, str) else meta
    except (TypeError, json.JSONDecodeError):
        return []

    label_ids: List[int] = []
    if meta_dict.get("label_set_id"):
        label_set = api.get_label_set(meta_dict["label_set_id"], project_id)
        if label_set and "relationships" in label_set:
            for item in label_set["relationships"].get("label_set_labels") or []:
                lid = item.get("label_id")
                if lid is not None:
                    label_ids.append(lid)
    # Worker uses train_params.label_ids from from_dict(meta); UI may send selected_label_ids
    if meta_dict.get("label_ids"):
        ids = meta_dict["label_ids"]
        label_ids.extend(ids if isinstance(ids, list) else [])
    elif meta_dict.get("selected_label_ids"):
        ids = meta_dict["selected_label_ids"]
        label_ids.extend(ids if isinstance(ids, list) else [])

    # crop_label_id: worker uses meta (or get_label when not in train_config); support top-level too
    crop_id = train_config.get("crop_label_id")
    if crop_id is None:
        crop_id = meta_dict.get("crop_label_id")
    if crop_id is not None and crop_id not in label_ids:
        label_ids.append(crop_id)

    return list(dict.fromkeys(label_ids))


def _resolve_meta_crop_label(api: Box21Api, project_id: int, train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve meta and ensure crop_label_id is set from crop_on_label if needed (matches worker get_train_params)."""
    meta = train_config.get("meta")
    if not meta:
        return {}
    try:
        meta_dict = json.loads(meta) if isinstance(meta, str) else meta
    except (TypeError, json.JSONDecodeError):
        return {}

    # Worker: if crop_label_id in train_config.keys() use meta's value, else get_label(crop_on_label)
    if meta_dict.get("crop_on_label") and meta_dict.get("crop_label_id") is None:
        if train_config.get("crop_label_id") is not None:
            meta_dict["crop_label_id"] = train_config["crop_label_id"]
        else:
            label = api.get_label(meta_dict["crop_on_label"], project_id)
            if label and "id" in label:
                meta_dict["crop_label_id"] = label["id"]
    elif train_config.get("crop_label_id") is not None:
        meta_dict["crop_label_id"] = train_config["crop_label_id"]
    return meta_dict


def get_annotations_for_train_config(
    api: Box21Api,
    project_id: int,
    train_config: Dict[str, Any],
    only_valid_assets: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch all annotations that would be used for this train config (train + validation).
    Matches box21_ai_worker BoundingBoxTrainer.load_annotations + Box21Dataset.get_annotations exactly:
    - Same label resolution (label_set_id, label_ids/selected_label_ids, crop_label_id)
    - Same filter building (base + train/val split + training_set_filters / validation_set_filters)
    - Same only_valid_assets default (True)
    - skip-fullsize-boxes: when config name contains "skip-fullsize-boxes", exclude coords [0,0,1,1]
    required_labels is NOT applied here (worker applies it when building crops, not in load_annotations).
    """
    train_label_ids = _resolve_train_label_ids(api, project_id, train_config)
    if not train_label_ids:
        return []

    labels = api.get_labels(label_ids=train_label_ids)
    if not labels:
        return []

    meta = _resolve_meta_crop_label(api, project_id, train_config)
    # Worker uses training_set_filters / validation_set_filters (plural); some APIs send singular
    train_filter_sets = _parse_filter_sets(meta.get("training_set_filters") or meta.get("training_set_filter"))
    val_filter_sets = _parse_filter_sets(meta.get("validation_set_filters") or meta.get("validation_set_filter"))

    annotations: List[Dict[str, Any]] = []
    for label in labels:
        for in_val, filter_sets in [(False, train_filter_sets), (True, val_filter_sets)]:
            anns = _fetch_annotations_for_label(
                api,
                project_id,
                label.name,
                in_validation_set=in_val,
                additional_filter_sets=filter_sets,
                only_valid_assets=only_valid_assets,
            )
            annotations.extend(anns)

    # skip-fullsize-boxes: worker filters these out when name contains "skip-fullsize-boxes"
    config_name = (train_config.get("name") or "") or ""
    if "skip-fullsize-boxes" in config_name:
        fullsize = "[0.0, 0.0, 1.0, 1.0]"
        annotations = [a for a in annotations if a.get("coords") != fullsize]

    return annotations


def _unique_assets(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe by asset_id, keep first occurrence (has filename etc.)."""
    seen: Set[int] = set()
    out = []
    for a in annotations:
        aid = a.get("asset_id")
        if aid is not None and aid not in seen:
            seen.add(aid)
            out.append(a)
    return out


def _annotations_by_asset(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group annotation records by asset_id."""
    by_asset: Dict[int, List[Dict[str, Any]]] = {}
    for a in annotations:
        aid = a.get("asset_id")
        if aid is None:
            continue
        if aid not in by_asset:
            by_asset[aid] = []
        by_asset[aid].append(
            {
                "label_id": a.get("label_id"),
                "label_name": a.get("label_name"),
                "coords": a.get("coords"),
            }
        )
    return by_asset


# --- Crop mode: "Only annotations contained in" + "Required labels in crop" ---

def _parse_coords(ann: Dict[str, Any]) -> Optional[List[float]]:
    """Return normalized coords as list of floats, or None."""
    raw = ann.get("coords")
    if raw is None:
        return None
    try:
        c = json.loads(raw) if isinstance(raw, str) else raw
        return [float(x) for x in c]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _in_box(needle: Dict[str, Any], haystack: Dict[str, Any]) -> bool:
    """True if needle (box) is fully contained in haystack (crop). Coords normalized 0-1 (x,y,w,h) or (x,y)."""
    nc = _parse_coords(needle)
    hc = _parse_coords(haystack)
    if not nc or not hc or len(hc) < 2:
        return False
    x, y = hc[0], hc[1]
    w = hc[2] if len(hc) > 2 else 0.0
    h = hc[3] if len(hc) > 3 else 0.0
    if len(nc) == 2:
        nx, ny = nc[0], nc[1]
        return x <= nx <= x + w and y <= ny <= y + h
    nx, ny, nw, nh = nc[0], nc[1], nc[2], nc[3]
    return not (nx > x + w or ny > y + h or nx + nw < x or ny + nh < y)


def _all_present(contained: List[Dict[str, Any]], required_labels: str) -> bool:
    """True if for each required label (comma-separated) there is at least one annotation whose label_name contains it."""
    want = [s.strip() for s in (required_labels or "").split(",") if s.strip()]
    if not want:
        return True
    present: Set[str] = set()
    for ann in contained:
        name = (ann.get("label_name") or "") or ""
        for r in want:
            if r in name:
                present.add(r)
    return len(present) == len(want)


def _transpose_to_crop(box_coords: List[float], crop_coords: List[float]) -> List[float]:
    """Transpose box from image-normalized to crop-normalized (0-1 relative to crop)."""
    if len(crop_coords) < 4:
        return box_coords
    cx, cy, cw, ch = crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3]
    if cw <= 0 or ch <= 0:
        return box_coords
    if len(box_coords) == 2:
        bx, by = box_coords[0], box_coords[1]
        return [(bx - cx) / cw, (by - cy) / ch]
    bx, by, bw, bh = box_coords[0], box_coords[1], box_coords[2], box_coords[3]
    return [
        (bx - cx) / cw,
        (by - cy) / ch,
        bw / cw,
        bh / ch,
    ]


def _is_crop_annotation(ann: Dict[str, Any], crop_on_label: str, crop_label_id: Optional[int]) -> bool:
    """True if this annotation defines a crop region (e.g. Spuitje or child of crop label)."""
    name = (ann.get("label_name") or "") or ""
    parent_id = ann.get("label_parent_id")
    if crop_on_label and name == crop_on_label:
        return True
    if crop_label_id is not None and parent_id == crop_label_id:
        return True
    if crop_on_label == "Spuitje" and "generated-spuitje" in name:
        return True
    if crop_on_label == "Amount" and (
        "i-amount" in name or "Amount" in name or "generated-amount" in name
    ):
        return True
    return False


def _crop_name(asset_id: int, crop_coords: List[float], crop_margin: int) -> str:
    """Stable name for a crop (matches worker-style)."""
    coord_str = "_".join(str(c).replace(".", "_") for c in crop_coords)
    return f"{asset_id}_{coord_str}_{crop_margin}"


def compute_valid_crops(
    annotations: List[Dict[str, Any]],
    crop_on_label: Optional[str],
    crop_label_id: Optional[int],
    required_labels: str,
    crop_margin: int = 10,
) -> List[Dict[str, Any]]:
    """
    When crop_on_label is set: return list of valid crops. Each item has:
    - crop_ann, asset_id, in_validation_set, filename, crop_name, contained (list of {label_id, label_name, coords} in crop coords).
    When crop_on_label is not set: return [] (no crop mode).
    """
    if not crop_on_label and crop_label_id is None:
        return []

    by_asset: Dict[int, List[Dict[str, Any]]] = {}
    for a in annotations:
        aid = a.get("asset_id")
        if aid is None:
            continue
        if aid not in by_asset:
            by_asset[aid] = []
        by_asset[aid].append(a)

    valid: List[Dict[str, Any]] = []
    for asset_id, anns in by_asset.items():
        crop_candidates = [a for a in anns if _is_crop_annotation(a, crop_on_label, crop_label_id)]
        boxes = [a for a in anns if a not in crop_candidates]
        for crop_ann in crop_candidates:
            crop_coords = _parse_coords(crop_ann)
            if not crop_coords or len(crop_coords) < 2:
                continue
            contained = [b for b in boxes if _in_box(b, crop_ann)]
            if not contained:
                continue
            if required_labels and required_labels.strip():
                if "generated" in (crop_ann.get("label_name") or ""):
                    pass
                elif not _all_present(contained, required_labels):
                    continue
            # Contained in image coords; we transpose when writing (need image size for point margin)
            if len(crop_coords) < 4:
                cx, cy = crop_coords[0], crop_coords[1]
                crop_coords = [cx, cy, 0.001, 0.001]
            contained_list = [
                {"label_id": b.get("label_id"), "label_name": b.get("label_name"), "coords": b.get("coords")}
                for b in contained
            ]
            crop_name = _crop_name(asset_id, crop_coords, crop_margin)
            valid.append({
                "crop_ann": crop_ann,
                "asset_id": asset_id,
                "in_validation_set": crop_ann.get("in_validation_set", False),
                "filename": crop_ann.get("filename") or f"{asset_id}.jpg",
                "crop_name": crop_name,
                "crop_coords": crop_coords,
                "contained": contained_list,
            })
    return valid


def _crop_bounds_and_normalized(
    img_width: int, img_height: int, crop_coords: List[float], crop_margin: int
) -> tuple:
    """Return (left, top, right, bottom) in pixels and normalized crop box [cx,cy,cw,ch] for transposing."""
    if len(crop_coords) >= 4:
        x, y, w, h = crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3]
        left = max(0, int(x * img_width))
        top = max(0, int(y * img_height))
        right = min(img_width, int((x + w) * img_width))
        bottom = min(img_height, int((y + h) * img_height))
        cx = left / img_width
        cy = top / img_height
        cw = (right - left) / img_width
        ch = (bottom - top) / img_height
        return (left, top, right, bottom), [cx, cy, cw, ch]
    # Point: expand by margin
    x, y = crop_coords[0], crop_coords[1]
    px = x * img_width
    py = y * img_height
    left = max(0, int(px - crop_margin))
    top = max(0, int(py - crop_margin))
    right = min(img_width, int(px + crop_margin))
    bottom = min(img_height, int(py + crop_margin))
    cx = left / img_width
    cy = top / img_height
    cw = (right - left) / img_width
    ch = (bottom - top) / img_height
    return (left, top, right, bottom), [cx, cy, cw, ch]


T = TypeVar("T")
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and response.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, (requests.ConnectionError, requests.Timeout, ConnectionResetError)):
        return True
    # Rate-limit responses often come back as non-image payloads.
    if isinstance(exc, (OSError, ValueError, json.JSONDecodeError)):
        return True
    return False


def _call_with_retry(fn: Callable[[], T], max_attempts: int = 6, base_delay: float = 1.0) -> T:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts - 1 or not _is_retryable_error(exc):
                raise
            time.sleep(base_delay * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


def _asset_image_path(asset_id: int, filename: str, assets_dir: Path) -> Path:
    ext = (filename or "jpg").split(".")[-1]
    if ext.lower() == "file":
        ext = "jpg"
    return assets_dir / f"{asset_id}.{ext}"


def _download_one_asset(
    api: Box21Api,
    asset_id: int,
    filename: str,
    assets_dir: Path,
    annotations_dir: Path,
    annotation_records: List[Dict[str, Any]],
    max_attempts: int = 6,
) -> bool:
    """Download one asset image and write its annotations JSON. Returns True if image saved."""
    image_path = _asset_image_path(asset_id, filename, assets_dir)
    if image_path.exists() and image_path.stat().st_size > 1000:
        return True

    try:
        img = _call_with_retry(lambda: api.download_asset(asset_id), max_attempts=max_attempts)
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(image_path, "JPEG", quality=95, optimize=True)
    except Exception:
        return False

    ann_file = annotations_dir / f"{asset_id}.json"
    with open(ann_file, "w") as f:
        json.dump(annotation_records, f, indent=2)
    return True


def _download_assets_parallel(
    download_jobs: List[Dict[str, Any]],
    max_workers: int,
    max_attempts: int,
    retry_failed_sequentially: bool = True,
) -> Set[int]:
    """Run asset downloads in parallel and optionally retry failures one-by-one."""
    downloaded: Set[int] = set()
    failed_jobs: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_one_asset,
                job["api"],
                job["asset_id"],
                job["filename"],
                job["assets_dir"],
                job["annotations_dir"],
                job["annotation_records"],
                max_attempts,
            ): job
            for job in download_jobs
        }
        progress = tqdm(as_completed(futures), total=len(futures), desc="Downloading assets", unit="asset")
        for future in progress:
            job = futures[future]
            try:
                if future.result():
                    downloaded.add(job["asset_id"])
                else:
                    failed_jobs.append(job)
            except Exception:
                failed_jobs.append(job)
            progress.set_postfix(ok=len(downloaded), failed=len(failed_jobs), refresh=False)

    if retry_failed_sequentially and failed_jobs:
        print(f"Retrying {len(failed_jobs)} failed asset downloads sequentially...")
        retry_progress = tqdm(failed_jobs, desc="Retrying failed assets", unit="asset")
        still_failed = 0
        for job in retry_progress:
            if _download_one_asset(
                job["api"],
                job["asset_id"],
                job["filename"],
                job["assets_dir"],
                job["annotations_dir"],
                job["annotation_records"],
                max_attempts=max_attempts,
            ):
                downloaded.add(job["asset_id"])
            else:
                still_failed += 1
            retry_progress.set_postfix(ok=len(downloaded), failed=still_failed, refresh=False)
            time.sleep(0.1)

    attempted = len(download_jobs)
    failed_count = attempted - len(downloaded)
    if failed_count:
        print(
            f"Asset download summary: {len(downloaded)}/{attempted} succeeded, {failed_count} failed.",
            file=sys.stderr,
        )
        print(
            "Re-run the same command to resume; existing files are skipped. "
            "Try a lower --max-workers if failures persist.",
            file=sys.stderr,
        )
    else:
        print(f"Asset download summary: {len(downloaded)}/{attempted} succeeded.")

    return downloaded


def download_assets_for_dataset(
    api: Box21Api,
    annotations: List[Dict[str, Any]],
    export_dir: Path,
    max_workers: int = 10,
    max_attempts: int = 6,
) -> Set[int]:
    """Download each unique asset image to export_dir/assets. Returns set of downloaded asset ids."""
    assets_dir = export_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = export_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    asset_list = _unique_assets(annotations)
    by_asset = _annotations_by_asset(annotations)

    download_jobs = [
        {
            "api": api,
            "asset_id": ann["asset_id"],
            "filename": ann.get("filename") or "jpg",
            "assets_dir": assets_dir,
            "annotations_dir": annotations_dir,
            "annotation_records": by_asset.get(ann["asset_id"], []),
        }
        for ann in asset_list
    ]
    return _download_assets_parallel(download_jobs, max_workers, max_attempts)


def download_assets_for_crop_dataset(
    api: Box21Api,
    valid_crops: List[Dict[str, Any]],
    export_dir: Path,
    max_workers: int = 10,
    max_attempts: int = 6,
) -> Set[int]:
    """Download only assets that have at least one valid crop. Saves to export_dir/assets."""
    asset_ids = list(dict.fromkeys(vc["asset_id"] for vc in valid_crops))
    if not asset_ids:
        return set()
    # Build flat annotations per asset for _download_one_asset (we only need asset image; annotations per asset not needed for crop mode)
    by_asset: Dict[int, Dict[str, Any]] = {}
    for vc in valid_crops:
        aid = vc["asset_id"]
        if aid not in by_asset:
            by_asset[aid] = {"asset_id": aid, "filename": vc["filename"], "annotations": []}
    assets_dir = export_dir / "assets"
    annotations_dir = export_dir / "annotations"
    assets_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    download_jobs = [
        {
            "api": api,
            "asset_id": info["asset_id"],
            "filename": info["filename"],
            "assets_dir": assets_dir,
            "annotations_dir": annotations_dir,
            "annotation_records": info["annotations"],
        }
        for info in by_asset.values()
    ]
    return _download_assets_parallel(download_jobs, max_workers, max_attempts)


def write_assets_metadata(
    api: Box21Api,
    asset_ids: Set[int],
    annotations: List[Dict[str, Any]],
    export_dir: Path,
    max_workers: int = 10,
    max_attempts: int = 6,
) -> None:
    """Fetch per-asset metadata and write export_dir/assets.json."""
    by_asset_hint = {a.get("asset_id"): a for a in _unique_assets(annotations) if a.get("asset_id") is not None}
    assets_out: List[Dict[str, Any]] = []
    failed_ids: List[int] = []

    def _fetch_one(asset_id: int) -> Optional[Dict[str, Any]]:
        try:
            asset = _call_with_retry(lambda: api.get_asset(asset_id), max_attempts=max_attempts)
        except Exception:
            return None
        raw_meta = getattr(asset, "meta", None)
        parsed_meta: Any = raw_meta
        if isinstance(raw_meta, str):
            try:
                parsed_meta = json.loads(raw_meta)
            except json.JSONDecodeError:
                parsed_meta = raw_meta
        hint = by_asset_hint.get(asset_id, {})
        return {
            "asset_id": asset_id,
            "filename": hint.get("filename"),
            "path": getattr(asset, "path", None),
            "project_id": getattr(asset, "project_id", None),
            "validated": getattr(asset, "validated", None),
            "in_validation_set": getattr(asset, "in_validation_set", None),
            "unclear": getattr(asset, "unclear", None),
            "liked": getattr(asset, "liked", None),
            "original_category": getattr(asset, "original_category", None),
            "meta": parsed_meta,
        }

    ids = sorted(asset_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, aid): aid for aid in ids}
        progress = tqdm(as_completed(futures), total=len(futures), desc="Fetching asset meta", unit="asset")
        for future in progress:
            aid = futures[future]
            rec = future.result()
            if rec is not None:
                assets_out.append(rec)
            else:
                failed_ids.append(aid)
            progress.set_postfix(ok=len(assets_out), failed=len(failed_ids), refresh=False)

    if failed_ids:
        print(f"Retrying metadata for {len(failed_ids)} assets sequentially...")
        for aid in tqdm(failed_ids, desc="Retrying asset meta", unit="asset"):
            rec = _fetch_one(aid)
            if rec is not None:
                assets_out.append(rec)

    assets_out.sort(key=lambda x: x.get("asset_id") or 0)
    assets_json = export_dir / "assets.json"
    with open(assets_json, "w") as f:
        json.dump(assets_out, f, indent=2)
    print(f"Wrote {assets_json} ({len(assets_out)} assets)")


def write_crops_and_annotations(
    export_dir: Path,
    valid_crops: List[Dict[str, Any]],
    crop_margin: int,
) -> None:
    """For each valid crop: load asset image, crop to region (with margin), save crop image and annotations with transposed coords."""
    assets_dir = export_dir / "assets"
    crops_dir = export_dir / "crops"
    annotations_dir = export_dir / "annotations"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for vc in tqdm(valid_crops, desc="Writing crops", unit="crop"):
        asset_id = vc["asset_id"]
        crop_name = vc["crop_name"]
        crop_coords = vc["crop_coords"]
        contained = vc["contained"]
        filename = vc["filename"]
        image_path = _asset_image_path(asset_id, filename or "jpg", assets_dir)
        if not image_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        w, h = img.size
        (left, top, right, bottom), norm_box = _crop_bounds_and_normalized(w, h, crop_coords, crop_margin)
        if right <= left or bottom <= top:
            skipped += 1
            continue
        crop_im = img.crop((left, top, right, bottom))
        crop_im.save(crops_dir / f"{crop_name}.jpg", "JPEG", quality=95)
        transposed = []
        for rec in contained:
            bc = _parse_coords(rec)
            if not bc:
                continue
            tc = _transpose_to_crop(bc, norm_box)
            transposed.append({
                "label_id": rec.get("label_id"),
                "label_name": rec.get("label_name"),
                "coords": json.dumps(tc),
            })
        with open(annotations_dir / f"{crop_name}.json", "w") as f:
            json.dump(transposed, f, indent=2)
        written += 1

    if skipped:
        print(
            f"Crop summary: {written}/{len(valid_crops)} written, {skipped} skipped (missing source asset image).",
            file=sys.stderr,
        )
    else:
        print(f"Crop summary: {written}/{len(valid_crops)} written.")


# Colors for drawing annotations (distinct per label)
_PREVIEW_COLORS = [
    (255, 99, 71), (50, 205, 50), (65, 105, 225), (255, 215, 0),
    (255, 20, 147), (0, 206, 209), (255, 165, 0), (147, 112, 219),
    (60, 179, 113), (220, 20, 60), (0, 191, 255), (255, 69, 0),
]


def _color_for_label(label_name: Optional[str]) -> tuple:
    """Return a consistent RGB color for a label name."""
    if not label_name:
        return _PREVIEW_COLORS[0]
    idx = hash(label_name) % len(_PREVIEW_COLORS)
    return _PREVIEW_COLORS[idx]


def _draw_annotations_on_image(
    img: Image.Image,
    annotation_records: List[Dict[str, Any]],
) -> Image.Image:
    """Draw bounding boxes and keypoints on a copy of the image. Returns the new image."""
    out = img.copy()
    if img.mode != "RGB":
        out = out.convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size

    for rec in annotation_records:
        coords_raw = rec.get("coords")
        label_name = rec.get("label_name") or ""
        if not coords_raw:
            continue
        try:
            coords = json.loads(coords_raw) if isinstance(coords_raw, str) else coords_raw
            coords = [float(c) for c in coords]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        color = _color_for_label(label_name)

        if len(coords) == 4:
            # Normalized xywh -> pixel xyxy
            x, y, bw, bh = coords
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, min(w, h) // 300))
            if label_name:
                draw.text((x1, max(0, y1 - 16)), label_name, fill=color)
        elif len(coords) >= 2:
            # Point (x, y)
            px = int(coords[0] * w)
            py = int(coords[1] * h)
            r = max(3, min(w, h) // 100)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=2, fill=color)
            if label_name:
                draw.text((px + r + 2, py - 10), label_name, fill=color)

    return out


def generate_preview_images(export_dir: Path) -> int:
    """
    Create preview folder with images that have annotations drawn on them.
    If crops/ exists: use crops/ and annotations/ (crop_name). Else: use assets/ and annotations/ (asset_id).
    """
    preview_dir = export_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = export_dir / "annotations"
    if not annotations_dir.exists():
        return 0

    count = 0
    # Prefer crops if present (crop mode)
    images_dir = export_dir / "crops" if (export_dir / "crops").exists() else export_dir / "assets"
    if not images_dir.exists():
        return 0
    for ann_path in annotations_dir.glob("*.json"):
        stem = ann_path.stem
        image_path = images_dir / f"{stem}.jpg"
        if not image_path.exists():
            for p in images_dir.glob(f"{stem}.*"):
                image_path = p
                break
        if not image_path or not image_path.exists():
            continue
        with open(ann_path) as f:
            records = json.load(f)
        if not records:
            continue
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        out = _draw_annotations_on_image(img, records)
        out_path = preview_dir / f"{stem}.jpg"
        out.save(out_path, "JPEG", quality=90)
        count += 1
    return count


def _print_filtering_status(meta: Optional[Dict[str, Any]], crop_mode: bool = False) -> None:
    """Print whether 'Required labels in crop' and 'Only annotations contained in' are applied in this export."""
    required_labels = (meta or {}).get("required_labels") or ""
    required_labels_set = bool(required_labels and str(required_labels).strip())
    crop_on = (meta or {}).get("crop_on_label") or (meta or {}).get("crop_label")
    print("Filtering in this export:")
    if crop_mode:
        print("  - Only annotations contained in: applied (crop-on label)", repr(crop_on))
        print("  - Required labels in crop: applied" + (f" (required: {repr(required_labels.strip())})" if required_labels_set else " (none set)"))
    else:
        print("  - Required labels in crop: not applied (no crop mode)")
        if required_labels_set:
            print(f"    Config has required_labels = {repr(required_labels.strip())}")
        print("  - Only annotations contained in: not applied (no crop mode)")


def _filtering_info_from_meta(meta: Optional[Dict[str, Any]], crop_mode: bool = False) -> Dict[str, Any]:
    """Build filtering section for summary."""
    required_labels = (meta or {}).get("required_labels") or ""
    required_labels_set = bool(required_labels and str(required_labels).strip())
    crop_on = (meta or {}).get("crop_on_label") or (meta or {}).get("crop_label")
    return {
        "required_labels_in_crop": {
            "config_value": required_labels.strip() if required_labels_set else None,
            "applied_in_this_export": crop_mode,
            "note": "Applied when building crops (all_present)." if crop_mode else "Not applied (no crop mode).",
        },
        "only_annotations_contained_in": {
            "config_value": crop_on,
            "applied_in_this_export": crop_mode,
            "note": "Crop region filter (in_box) applied; only annotations inside crop kept." if crop_mode else "Not applied (no crop mode).",
        },
    }


def write_dataset_manifest(
    export_dir: Path,
    annotations: List[Dict[str, Any]],
    train_config_id: int,
    project_id: int,
    train_config_name: str,
    train_config_meta: Optional[Dict[str, Any]] = None,
    valid_crops: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write labels.txt, train.txt, valid.txt and dataset_summary.json. If valid_crops given, use crop paths and labels from contained."""
    crop_mode = valid_crops is not None and len(valid_crops) > 0
    if crop_mode:
        all_labels = []
        for vc in valid_crops:
            for rec in vc.get("contained") or []:
                n = rec.get("label_name")
                if n:
                    all_labels.append(n)
        all_labels = list(dict.fromkeys(all_labels))
        all_labels.sort()
        train_crops = [vc for vc in valid_crops if not vc.get("in_validation_set")]
        valid_crops_list = [vc for vc in valid_crops if vc.get("in_validation_set")]
        train_paths = [f"crops/{vc['crop_name']}.jpg" for vc in train_crops]
        valid_paths = [f"crops/{vc['crop_name']}.jpg" for vc in valid_crops_list]
        num_annotations = sum(len(vc.get("contained") or []) for vc in valid_crops)
    else:
        all_labels = list(dict.fromkeys(a["label_name"] for a in annotations if a.get("label_name")))
        all_labels.sort()
        train_assets = [a for a in annotations if not a.get("in_validation_set")]
        valid_assets = [a for a in annotations if a.get("in_validation_set")]

        def asset_ids_to_paths(anns: List[Dict[str, Any]]) -> List[str]:
            seen: Set[int] = set()
            paths = []
            for a in anns:
                aid = a.get("asset_id")
                if aid is None or aid in seen:
                    continue
                seen.add(aid)
                fn = a.get("filename") or f"{aid}.jpg"
                ext = fn.split(".")[-1]
                if ext.lower() == "file":
                    ext = "jpg"
                paths.append(f"assets/{aid}.{ext}")
            return paths

        train_paths = asset_ids_to_paths(train_assets)
        valid_paths = asset_ids_to_paths(valid_assets)
        num_annotations = len(annotations)

    labels_file = export_dir / "labels.txt"
    with open(labels_file, "w") as f:
        f.write("\n".join(all_labels))

    train_file = export_dir / "train.txt"
    with open(train_file, "w") as f:
        f.write("\n".join(train_paths))

    valid_file = export_dir / "valid.txt"
    with open(valid_file, "w") as f:
        f.write("\n".join(valid_paths))

    summary = {
        "train_config_id": train_config_id,
        "project_id": project_id,
        "train_config_name": train_config_name,
        "num_train_assets": len(train_paths),
        "num_valid_assets": len(valid_paths),
        "num_annotations": num_annotations,
        "labels": all_labels,
        "crop_mode": crop_mode,
        "filtering": _filtering_info_from_meta(train_config_meta, crop_mode=crop_mode),
    }
    summary_file = export_dir / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {labels_file}, {train_file}, {valid_file}, {summary_file}")


def run_download(
    api: Box21Api,
    project_id: int,
    train_config_id: int,
    export_dir: Path,
    max_workers: int = 10,
    max_attempts: int = 6,
) -> bool:
    """Resolve config, fetch annotations, download assets, write manifest. Returns True on success."""
    train_config = api.get_train_config(train_config_id, project_id)
    if not train_config:
        print("Failed to load train config", file=sys.stderr)
        return False

    name = train_config.get("name") or f"config_{train_config_id}"
    print(f"Train config: {name} (id={train_config_id})")
    annotations = get_annotations_for_train_config(api, project_id, train_config)
    if not annotations:
        print("No annotations found for this config.", file=sys.stderr)
        return False

    # Resolve meta for summary and crop mode
    meta_raw = train_config.get("meta")
    try:
        train_config_meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
    except (TypeError, json.JSONDecodeError):
        train_config_meta = {}
    crop_on_label = train_config_meta.get("crop_on_label") or train_config_meta.get("crop_label")
    crop_label_id = train_config_meta.get("crop_label_id")
    required_labels = (train_config_meta.get("required_labels") or "").strip()
    raw_margin = train_config_meta.get("crop_margin", 10)
    try:
        crop_margin = int(raw_margin)
    except (TypeError, ValueError):
        try:
            crop_margin = int(float(str(raw_margin).split()[0]))
        except (TypeError, ValueError, IndexError):
            crop_margin = 10

    print(f"Found {len(annotations)} annotations")
    crop_mode = bool(crop_on_label or crop_label_id is not None)
    valid_crops: Optional[List[Dict[str, Any]]] = None
    downloaded_assets: Set[int] = set()
    if crop_mode:
        valid_crops = compute_valid_crops(
            annotations, crop_on_label, crop_label_id, required_labels, crop_margin
        )
        print(f"Valid crops (contained + required_labels): {len(valid_crops)}")
        if not valid_crops:
            print("No valid crops; nothing to export.", file=sys.stderr)
            return False
        _print_filtering_status(train_config_meta, crop_mode=True)
        downloaded_assets = download_assets_for_crop_dataset(
            api, valid_crops, export_dir, max_workers=max_workers, max_attempts=max_attempts
        )
        write_crops_and_annotations(export_dir, valid_crops, crop_margin)
    else:
        _print_filtering_status(train_config_meta, crop_mode=False)
        downloaded_assets = download_assets_for_dataset(
            api, annotations, export_dir, max_workers=max_workers, max_attempts=max_attempts
        )
    write_assets_metadata(
        api, downloaded_assets, annotations, export_dir,
        max_workers=max_workers, max_attempts=max_attempts,
    )
    preview_count = generate_preview_images(export_dir)
    if preview_count:
        print(f"Wrote {preview_count} preview images to {export_dir / 'preview'}")
    write_dataset_manifest(
        export_dir,
        annotations,
        train_config_id,
        project_id,
        name,
        train_config_meta=train_config_meta,
        valid_crops=valid_crops if crop_mode else None,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download dataset for a train config (same as box21_ai_worker would use for training)"
    )
    parser.add_argument(
        "train_config_id",
        type=int,
        nargs="?",
        default=None,
        help="Train config ID (optional: will list configs after selecting project)",
    )
    parser.add_argument("project_id", type=int, nargs="?", default=None, help="Project ID (optional if only one project)")
    parser.add_argument("--host", help="Box21 host URL")
    parser.add_argument("--username", help="Box21 username/email")
    parser.add_argument("--export-dir", help="Output directory (default: ~/.box21_exports/datasets/<project_id>_<train_config_id>)")
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel download workers (default: 5)")
    parser.add_argument("--retry-attempts", type=int, default=6, help="Retry attempts for rate-limited API calls (default: 6)")

    args = parser.parse_args()

    host = args.host or get_user_input("Enter Box21 host", "https://box21.ai")
    username = args.username or get_user_input("Enter username/email")
    password = get_password("Enter password")

    if ":" in host and not host.startswith("http"):
        host_parts = host.split(":")
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) == 2 else 443
    else:
        port = 443 if host.startswith("https") else 80
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"

    api = Box21Api(email=username, password=password, host=host, port=port, project_id=0)

    project_id = args.project_id
    if project_id is None:
        projects = api.get_projects()
        if not projects:
            print("No projects found.", file=sys.stderr)
            return 1
        selected = select_from_list(projects, "name")
        project_id = selected.id
        print(f"\nSelected project: {selected.name} (ID: {project_id})")
    api.project_id = project_id

    train_config_id = args.train_config_id
    if train_config_id is None:
        print("\nFetching train configs...")
        train_configs = api.get_train_configs(project_id)
        if not train_configs:
            print("No train configs found or list not supported. Enter train config ID manually.")
            while True:
                try:
                    train_config_id = int(get_user_input("Train config ID"))
                    break
                except ValueError:
                    print("Please enter a valid number.")
        else:
            selected_config = select_from_list(train_configs, "name")
            train_config_id = (
                selected_config.get("id")
                if isinstance(selected_config, dict)
                else getattr(selected_config, "id", None)
            )
            print(f"\nSelected train config ID: {train_config_id}")

    export_dir = Path(args.export_dir) if args.export_dir else (
        Path.home() / ".box21_exports" / "datasets" / f"{project_id}_{train_config_id}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Export directory: {export_dir}")

    if run_download(api, project_id, train_config_id, export_dir, args.max_workers, args.retry_attempts):
        print("Dataset download completed successfully.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
