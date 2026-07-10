"""Interact with your Box21"""

# %% auto 0
__all__ = ["Box21Api"]

from fastcore.utils import *  # type: ignore

from typing import Any, Optional, List, Dict
import threading
import requests


class Box21Api:
    "Box21 Api class"

    def __init__(
        self, email: str, password: str, host: str, port: int, project_id: int
    ):
        self.email = email
        self.password = password
        self.host = host
        self.port = port
        self.project_id = project_id
        self._token_lock = threading.Lock()
        self.token: Optional[str] = None
        self.refresh_token()

    def _url(self, path: str) -> str:
        return f"{self.host}:{self.port}{path}"

    def _auth_headers(self) -> Dict[str, str]:
        token = self.get_token()
        return {"x-access-token": token}

    def refresh_token(self) -> str:
        with self._token_lock:
            payload = {"email": self.email, "password": self.password}
            response = requests.post(self._url("/api/login"), data=payload)
            response.raise_for_status()
            token = response.json().get("token")
            if not token:
                raise ValueError("Login did not return a token")
            self.token = token
            return token

    def get_token(self) -> str:
        if not self.token:
            return self.refresh_token()
        return self.token

    def post(
        self, url: str, payload: Dict[str, Any], files: Optional[Dict[str, str]] = None
    ):
        return self._request("post", url, payload=payload, files=files)

    def get(self, url: str, payload: Optional[Dict[str, str]] = None):
        return self._request("get", url, payload=payload)

    def _request(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, str]] = None,
        _retried: bool = False,
    ):
        headers = self._auth_headers()
        full_url = self._url(url)
        if method == "post":
            if files:
                response = requests.post(full_url, data=payload, files=files, headers=headers)
            else:
                response = requests.post(full_url, data=payload, headers=headers)
        else:
            response = requests.get(full_url, data=payload, headers=headers)

        if response.status_code == 401 and not _retried:
            self.refresh_token()
            return self._request(method, url, payload=payload, files=files, _retried=True)
        return response


from .asset import Asset


@patch
def get_assets(
    self: Box21Api, offset: int = 0, limit: Optional[int] = None
) -> List[Asset]:
    payload: Dict[str, str] = {
        "offset": str(offset),
        "limit": limit,  # type: ignore
        "project_id": str(self.project_id),
    }
    url = "/api/v2/assets"
    response = self.post(url, payload)

    asset_jsons = response.json()

    return [Asset.from_json(asset_json) for asset_json in asset_jsons]


@patch
def get_asset(self: Box21Api, asset_id: int) -> Asset:
    url = "/api/asset"
    response = self.post(url, {"asset_id": asset_id})
    response.raise_for_status()
    asset_json = response.json()
    return Asset.from_json(asset_json)


from PIL import Image
import io


@patch
def download_asset(self: Box21Api, asset_id: int) -> Image.Image:
    url = "/api/asset/download"
    payload = {"asset_id": asset_id, "project_id": self.project_id}
    response = self.post(url, payload)
    response.raise_for_status()

    return Image.open(io.BytesIO(response.content))


import json


@patch
def get_assets_with_filters(
    self: Box21Api, filters: List[Dict[str, str]], offset: int = 0, limit: int = 100
) -> List[Asset]:
    api_endpoint = "/api/filter-assets"

    form_params: Dict[str, Any] = {
        "filters": json.dumps(filters),
        "project_id": self.project_id,
        "offset": offset,
        "limit": limit,
    }

    response = self.post(api_endpoint, form_params)

    response_data = response.json()
    items = response_data.get("items", [])
    asset_ids = [item["id"] for item in items]

    # Second request to get asset details
    api_endpoint = "/api/v2/assets"
    form_params = {
        "asset_ids": json.dumps(asset_ids),
        "project_id": self.project_id,
        "limit": len(asset_ids),
    }

    response = self.post(api_endpoint, form_params)

    return [Asset.from_json(asset_json) for asset_json in response.json()]


@patch
def get_assets_with_filters_v2(
    self: Box21Api, filters: List[Dict[str, str]], offset: int = 0, limit: int = 100
) -> List[Asset]:
    api_endpoint = "/api/assets/filter/v2"

    form_params: Dict[str, Any] = {
        "filters": json.dumps(filters),
        "project_id": self.project_id,
        "offset": offset,
        "limit": limit,
    }

    response = self.post(api_endpoint, form_params)

    return [Asset.from_json(asset_json) for asset_json in response.json()]


@patch
def get_assets_containing_meta(
    self: Box21Api,
    meta: List[str],
    in_validation_set: bool = False,
    validated: Optional[bool] = None,
) -> List[Asset]:
    filters: List[Dict[str, Any]] = []

    for m in meta:
        filters.append({"type": 7, "value": m})

    if in_validation_set:
        filters.append({"type": 13, "value": "Yes"})
    else:
        filters.append({"type": 13, "value": "No"})

    if validated is True:
        filters.append({"type": 9, "value": "Yes"})
    elif validated is False:
        filters.append({"type": 9, "value": "No"})

    return self.get_assets_with_filters(filters)


@patch
def get_assets_not_containing_meta(
    self: Box21Api,
    meta: List[str],
    in_validation_set: bool = False,
    validated: Optional[bool] = None,
) -> List[Asset]:
    filters: List[Dict[str, Any]] = []

    for m in meta:
        filters.append({"type": 8, "value": m})

    if in_validation_set:
        filters.append({"type": 13, "value": "Yes"})
    else:
        filters.append({"type": 13, "value": "No"})

    if validated is True:
        filters.append({"type": 9, "value": "Yes"})
    elif validated is False:
        filters.append({"type": 9, "value": "No"})

    return self.get_assets_with_filters(filters)


from .annotation import Box21Annotation, Box21BoundingBox, Box21Keypoint
from .annotation import parse_json_annotation


@patch
def get_annotations(self: Box21Api, asset_id: int) -> List[Box21Annotation]:
    url = "/api/asset/annotations"
    payload = {
        "asset_id": asset_id,
    }
    response = self.post(url, payload)

    annotations: List[Box21Annotation] = []

    for annotation_json in response.json():
        annotations.append(parse_json_annotation(annotation_json))

    return annotations


@patch
def filter_boxes(
    self: Box21Api,
    filters: List[Dict[str, Any]],
    offset: int = 0,
    limit: int = 100,
    project_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Filter annotations using the /api/filter-boxes/v2 endpoint

    Args:
        filters: List of filter dictionaries to apply
        offset: Starting offset for pagination (default: 0)
        limit: Maximum number of results to return (default: 100)
        project_id: Project ID to filter on (defaults to self.project_id)

    Returns:
        Dictionary with 'annotations' and 'max_results' keys
    """
    if project_id is None:
        project_id = self.project_id

    url = "/api/filter-boxes/v2"
    payload = {
        "filters": json.dumps(filters),
        "project_id": str(project_id),
        "offset": str(offset),
        "limit": str(limit),
    }

    response = self.post(url, payload)
    return response.json()


@patch
def get_crop(
    self: Box21Api, asset_id: int, bbox: List[float], label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a cropped image using the /api/get_crop endpoint

    Args:
        asset_id: ID of the asset to crop from
        bbox: Bounding box coordinates as [x, y, w, h] (normalized 0-1) or [x, y] for keypoints
        label: Optional label name to include in the response

    Returns:
        Dictionary with 'crops' key containing list of crop data with 'crop' (base64) and 'label' fields
    """
    url = "/api/get_crop"
    payload = {"asset_id": str(asset_id), "bbox": json.dumps(bbox)}

    if label is not None:
        payload["label"] = label

    response = self.post(url, payload)

    return response.json()


@patch
def update_asset_meta(
    self: Box21Api, asset_id: int, key: str, value: str
) -> [Box21Annotation]:
    url = "/api/asset/meta/update-value"
    payload = {"asset_id": asset_id, "key": key, "value": value}
    response = self.post(url, payload)

    return Asset.from_json(response.json())


@patch
def delete_asset_meta_key(self: Box21Api, asset_id: int, key: str) -> [Box21Annotation]:
    url = "/api/asset/meta/delete-key"
    payload = {"asset_id": asset_id, "key": key}
    response = self.post(url, payload)

    return Asset.from_json(response.json())


from .label import Box21Label
from typing import List
from .annotation_stat import AnnotationStat
from .project import Project


@patch
def get_train_configs(self: Box21Api, project_id: int) -> List[Dict[str, Any]]:
    """List train configs for a project. Returns list of dicts with at least 'id' and 'name'."""
    try:
        payload = {"project_id": project_id}
        response = self.post("/api/train-configs", payload)
        if response is None or response.status_code != 200:
            return []
        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "train_configs" in data:
            return data["train_configs"]
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        return []
    except Exception:
        return []


@patch
def get_train_config(self: Box21Api, config_id: int, project_id: int) -> Optional[Dict[str, Any]]:
    """Fetch train config by id and project."""
    url = "/api/train-config"
    payload = {"config_id": config_id, "project_id": project_id}
    response = self.post(url, payload)
    if response is None or response.status_code != 200:
        return None
    return response.json()


@patch
def get_label_set(self: Box21Api, label_set_id: int, project_id: int) -> Optional[Dict[str, Any]]:
    """Fetch label set by id and project."""
    url = f"/api/label_set/{label_set_id}"
    payload = {"project_id": project_id}
    response = self.post(url, payload)
    if response is None or response.status_code != 200:
        return None
    return response.json()


@patch
def get_label(self: Box21Api, label_name: str, project_id: int, type: int = 1) -> Optional[Dict[str, Any]]:
    """Fetch label by name and project (e.g. for crop_on_label)."""
    url = "/api/label/get-label-id"
    payload = {"project_id": project_id, "name": label_name, "type": type}
    response = self.post(url, payload)
    if response is None or response.status_code != 200:
        return None
    return response.json()


@patch
def get_labels(self: Box21Api, label_ids: Optional[List[int]] = None) -> List[Box21Label]:
    url = "/api/labels"
    payload: Dict[str, Any] = {"project_id": self.project_id}
    if label_ids is not None:
        payload["label_ids"] = json.dumps(label_ids)
    response = self.post(url, payload)
    if response is None or response.status_code != 200:
        return []

    labels: List[Box21Label] = []
    for label_json in response.json():
        annotation_stats: List[AnnotationStat] = []

        for annotation_stat_json in label_json["relationships"]["annotation_stats"]:
            annotation_stats.append(AnnotationStat.from_json(annotation_stat_json))

        labels.append(
            Box21Label(
                id=label_json["id"],
                name=label_json["name"],
                parent_id=label_json["parent_id"],
                project_id=label_json["project_id"],
                type=label_json["type"],
                annotation_stats=annotation_stats,
            )
        )

    return labels


@patch
def get_label_annotations(self: Box21Api, label: Box21Label) -> [Box21Annotation]:
    url = "/api/label/annotations"
    payload = {"label_id": label.id}
    response = self.post(url, payload)

    annotations = []

    for annotation_json in response.json():
        asset_id = annotation_json["asset_id"]
        annotation_id = annotation_json["id"]
        certainty = annotation_json["certainty"]
        label_id = annotation_json["label_id"]
        project_id = annotation_json["project_id"]
        validated = annotation_json["validated"]
        coords = json.loads(annotation_json["coords"])

        if annotation_json["type"] == 1:
            x, y, w, h = coords
            annotations.append(
                Box21BoundingBox(
                    asset_id,
                    annotation_id,
                    certainty,
                    label_id,
                    project_id,
                    validated,
                    x,
                    y,
                    w,
                    h,
                )
            )
        else:
            x, y = coords
            annotations.append(
                Box21Keypoint(
                    asset_id,
                    annotation_id,
                    certainty,
                    label_id,
                    project_id,
                    validated,
                    x,
                    y,
                )
            )
    return annotations


from typing import Any, Dict, List
import json


@patch
def re_label(
    self: Box21Api, annotations: List[Box21Annotation], label_name: str
) -> None:
    url = "/api/re-label"
    annotation_ids: List[int] = [annotation.id for annotation in annotations]
    payload: Dict[str, Any] = {
        "annotation_ids": json.dumps(annotation_ids),
        "label": label_name,
    }
    self.post(url, payload)


@patch
def rename_label(self: Box21Api, label: Box21Label, new_name: str) -> None:
    """Rename a label"""
    url = "/api/update-label"
    payload: Dict[str, Any] = {"id": label.id, "name": new_name}
    self.post(url, payload)


from pathlib import Path
from .annotation import Annotation
from .annotation import BoundingBox
from .annotation import Keypoint


@patch
def add_asset(
    self: Box21Api,
    file_path: Path,
    meta: Dict[str, Any],
    annotations: List[Annotation] = [],
    validated: bool = False,
    in_validation_set: bool = False,
    no_duplicate_filename: bool = False,
) -> List[Asset]:
    if not isinstance(meta, dict):
        return "meta argument should be a python dictionary"

    bounding_boxes: List[BoundingBox] = []
    keypoints: List[Keypoint] = []
    for annotation in annotations:
        if isinstance(annotation, BoundingBox):
            if annotation.x > 1:
                return "Incorrect coordinates, should be between 0 and 1"
            bounding_boxes.append(
                {
                    "normalized_xywh": [
                        annotation.x,
                        annotation.y,
                        annotation.width,
                        annotation.height,
                    ],
                    "label": annotation.label_name,
                    "confidence": annotation.certainty,
                }
            )
        elif isinstance(annotation, Keypoint):
            if annotation.x > 1:
                return "Incorrect coordinates, should be between 0 and 1"
            keypoints.append(
                {
                    "normalized_xywh": [annotation.x, annotation.y],
                    "label": annotation.label_name,
                    "confidence": annotation.certainty,
                }
            )

    url = "/api/assets/add"
    payload = {
        "meta": json.dumps(meta),
        "validated": validated,
        "in_validation_set": in_validation_set,
        "filename": file_path.name,
        "bounding_boxes": json.dumps(bounding_boxes),
        "keypoints": json.dumps(keypoints),
        "project_id": self.project_id,
        "no_duplicate_filename": no_duplicate_filename,
    }
    files = {"file": open(file_path, "rb")}
    response = self.post(url, payload, files=files)
    # print(response.text)

    return Asset.from_json(response.json())


@patch
def delete_assets(self: Box21Api, asset_ids: List[int]):
    url = "/api/assets/delete"
    payload = {"asset_ids": json.dumps(asset_ids)}
    response = self.post(url, payload)
    return response


@patch
def invalidate_assets(self: Box21Api, asset_ids: List[int]):
    url = "/api/assets/invalidate"
    payload = {"asset_ids": json.dumps(asset_ids)}
    response = self.post(url, payload)
    return response


from requests import Response


@patch
def add_annotation(
    self: Box21Api, annotation: Annotation, asset: Asset
) -> Optional[Response]:
    # is BoundingBox type
    if type(annotation) == BoundingBox:
        url = "/api/assets/boundingbox/add"
        label_name = annotation.label_name
        normalized_xywh = [
            annotation.x,
            annotation.y,
            annotation.width,
            annotation.height,
        ]
        certainty = annotation.certainty
        payload: Dict[str, Any] = {
            "asset_id": asset.id,
            "label_name": label_name,
            "normalized_xywh": json.dumps(normalized_xywh),
            "certainty": certainty,
        }

        return self.post(url, payload)

    elif type(annotation) == Keypoint:
        url = "/api/assets/keypoint/add"
        label_name = annotation.label_name
        normalized_xy = [annotation.x, annotation.y]
        certainty = annotation.certainty
        payload: Dict[str, Any] = {
            "asset_id": asset.id,
            "label_name": label_name,
            "normalized_xywh": json.dumps(normalized_xy),
            "certainty": certainty,
        }
        return self.post(url, payload)


@patch
def detete_annotations(self: Box21Api, annotations: List[Box21Annotation]) -> None:
    url = "/api/annotation/delete"
    for annotation in annotations:
        payload = {"annotation_id": annotation.id}
        self.post(url, payload)


@patch
def update_job(
    self: Box21Api,
    job_id: int,
    processing: Optional[bool] = None,
    processed: Optional[bool] = None,
    progress: Optional[int] = None,
):
    url = "/api/update-job"
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "processing": processing,
        "processed": processed,
        "progress": progress,
    }
    return self.post(url, payload)


from requests import Response


@patch
def delete_suggestions(self: Box21Api, asset_id: int) -> Response:
    url = "/api/asset/discard-all-suggestions"
    payload = {"asset_id": asset_id}
    return self.post(url, payload)


@patch
def get_models(self: Box21Api, project_id: int) -> List[Dict[str, Any]]:
    """Get all models for a specific project."""
    try:
        payload = {"project_id": project_id}
        response = self.post("/api/models", payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


@patch
def get_projects(self: Box21Api) -> List[Project]:
    """Get all projects for the authenticated user."""
    try:
        # Use the correct API endpoint for getting projects
        response = self.post("/api/projects", {})
        if response.status_code == 200:
            project_jsons = response.json()
            return [Project.from_json(project_json) for project_json in project_jsons]

        # Fallback: try GET method if POST doesn't work
        response = self.get("/api/projects")
        if response.status_code == 200:
            project_jsons = response.json()
            return [Project.from_json(project_json) for project_json in project_jsons]

        # If still no success, try a different endpoint
        response = self.get("/api/user/projects")
        if response.status_code == 200:
            project_jsons = response.json()
            return [Project.from_json(project_json) for project_json in project_jsons]

        # If still no success, try to get projects by attempting to get labels
        # and extracting project info from the first successful response
        projects = []
        for project_id in range(1, 100):  # Try first 100 project IDs
            try:
                # Temporarily set project_id
                original_project_id = self.project_id
                self.project_id = project_id

                # Try to get labels for this project
                labels = self.get_labels()
                if labels:
                    # If we get labels, this project exists
                    project_info = {
                        "id": project_id,
                        "name": f"Project {project_id}",
                        "size": 0,  # We don't know the actual size
                        "description": None,
                        "created_at": None,
                        "updated_at": None,
                        "deleted_at": None,
                        "owner_user_id": None,
                    }
                    projects.append(Project.from_json(project_info))

                # Restore original project_id
                self.project_id = original_project_id

            except Exception:
                # Project doesn't exist or we can't access it
                continue

        return projects

    except Exception as e:
        print(f"Warning: Could not get projects: {e}")
        return []
