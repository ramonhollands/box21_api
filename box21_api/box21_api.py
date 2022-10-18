# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_api.ipynb.

# %% auto 0
__all__ = ['Box21Api']

# %% ../01_api.ipynb 4
from fastcore.utils import *

# %% ../01_api.ipynb 5
import requests

class Box21Api:
    "Box21 Api class"
    def __init__(self, email:str, password:str, host:str, port:int, project_id:int):
        self.email = email
        self.password = password
        self.host = host
        self.port = port
        self.project_id = project_id
        self.token = self.get_token()
        
    def post(self, url, payload, files=None):
        self.token = self.get_token()
        session = requests.Session()
        session.headers.update({'x-access-token': self.token})
        if files:
            response = session.post(self.host + ':' + str(self.port) + url, data=payload, files=files)
        else:
            response = session.post(self.host + ':' + str(self.port) + url, data=payload)
        return response
    
    def get(self, url, payload=None):
        self.token = self.get_token()
        session = requests.Session()
        session.headers.update({'x-access-token': self.token})
        response = session.get(self.host + ':' + str(self.port) + url, data=payload)
        return response
        
    def get_token(self):
        session = requests.Session()

        # Get token
        payload = {
            "email": self.email, "password": self.password}

        response = session.post(self.host + ':' + str(self.port) + '/api/login', data=payload)
        token = response.json().get('token')
        return token

# %% ../01_api.ipynb 8
from .asset import Asset

@patch
def get_assets(self:Box21Api, offset:int=0, limit:int=None) -> [Asset]:
    payload = {
        "offset": offset,
        "limit": limit,
        "project_id": self.project_id
    }
    url = '/api/v2/assets'
    response = self.get(url, payload)
    asset_jsons = response.json()

    return [Asset.from_json(asset_json) for asset_json in asset_jsons]

@patch
def get_asset(self:Box21Api, asset_id: int) -> Asset:
    url = '/api/asset'
    response = self.post(url, {'asset_id': asset_id})
    asset_json = response.json()    
    return Asset.from_json(asset_json)

# %% ../01_api.ipynb 12
from PIL import Image
import io

@patch
def download_asset(self:Box21Api, asset_id: int) -> Image:
    self.token = self.get_token()
    url = '/api/asset/download'
    payload = {
        "asset_id": asset_id,
        "project_id": self.project_id
    }
    response = self.post(url, payload)

    return Image.open(io.BytesIO(response.content))

# %% ../01_api.ipynb 16
from PIL import Image
import io, json
from .annotation import Annotation, BoundingBox, Keypoint

@patch
def get_annotations(self:Box21Api, asset_id: int) -> [Annotation]:
    self.token = self.get_token()
    url = '/api/asset/annotations'
    payload = {
        "asset_id": asset_id,
    }
    response = self.post(url, payload)

    annotations = []

    for annotation_json in response.json():

        asset_id = annotation_json['asset_id']
        annotation_id = annotation_json['id']
        certainty = annotation_json['certainty']
        label_id = annotation_json['label_id']
        label_name = annotation_json['relationships']['label']['name']
        project_id = annotation_json['project_id']
        validated = annotation_json['validated']
        coords = json.loads(annotation_json['coords'])

        if annotation_json['type'] == 1:
            x, y, w, h = coords
            annotations.append(
                BoundingBox(asset_id, annotation_id, certainty, label_id, label_name, project_id, validated, x, y, w, h))
        else:
            x, y = coords
            annotations.append(
                Keypoint(asset_id, annotation_id, certainty, label_id, label_name, project_id, validated, x, y))


    return annotations

# %% ../01_api.ipynb 20
@patch
def update_asset_meta(self:Box21Api, asset_id: int, key: str, value: str) -> [Annotation]:
    self.token = self.get_token()
    url = '/api/asset/meta/update-value'
    payload = {
        "asset_id": asset_id,
        "key": key,
        "value": value
    }
    response = self.post(url, payload)

    return Asset.from_json(response.json())

@patch
def delete_asset_meta_key(self:Box21Api, asset_id: int, key: str) -> [Annotation]:
    self.token = self.get_token()
    url = '/api/asset/meta/delete-key'
    payload = {
        "asset_id": asset_id,
        "key": key
    }
    response = self.post(url, payload)

    return Asset.from_json(response.json())

# %% ../01_api.ipynb 26
from .label import Label

@patch
def get_labels(self:Box21Api) -> [Label]:
    self.token = self.get_token()
    url = '/api/labels'
    payload = {
        "project_id": self.project_id
    }
    response = self.post(url, payload)

    labels = []
    for label_json in response.json():
        labels.append(
            Label(
                id=label_json['id'],
                name=label_json['name'],
                parent_id=label_json['parent_id'],
                project_id=label_json['project_id'],
                type=label_json['type']
            ))

    return labels

# %% ../01_api.ipynb 29
@patch
def get_label_annotations(self:Box21Api, label: Label) -> [Annotation]:
    self.token = self.get_token()
    url = '/api/label/annotations'
    payload = {
        "label_id": label.id
    }
    response = self.post(url, payload)

    annotations = []

    for annotation_json in response.json():

        asset_id = annotation_json['asset_id']
        annotation_id = annotation_json['id']
        certainty = annotation_json['certainty']
        label_id = annotation_json['label_id']
        label_name = annotation_json['relationships']['label']['name']
        project_id = annotation_json['project_id']
        validated = annotation_json['validated']
        coords = json.loads(annotation_json['coords'])

        if annotation_json['type'] == 1:
            x, y, w, h = coords
            annotations.append(
                BoundingBox(asset_id, annotation_id, certainty, label_id, label_name, project_id, validated, x, y, w, h))
        else:
            x, y = coords
            annotations.append(
                Keypoint(asset_id, annotation_id, certainty, label_id, label_name, project_id, validated, x, y))
    return annotations

# %% ../01_api.ipynb 32
from pathlib import Path
from .annotation import Annotation
from .annotation import BoundingBox
from .annotation import Keypoint

@patch
def add_asset(self:Box21Api, file_path: Path, meta, annotations: [Annotation]= []) -> [Asset]:

    if not isinstance(meta, dict):
        return 'meta argument should be a python dictionary'

    bounding_boxes = []
    keypoints = []
    for annotation in annotations:
        if isinstance(annotation, BoundingBox):
            if annotation.x > 1:
                return 'Incorrect coordinates, should be between 0 and 1'
            bounding_boxes.append({
                'normalized_xywh': [annotation.x, annotation.y, annotation.width, annotation.height],
                'label': annotation.label_name,
                'confidence': annotation.certainty
            })
        elif isinstance(annotation, Keypoint):
            if annotation.x > 1:
                return 'Incorrect coordinates, should be between 0 and 1'
            keypoints.append({
                'normalized_xywh': [annotation.x, annotation.y],
                'label': annotation.label_name,
                'confidence': annotation.certainty
            })


    self.token = self.get_token()
    url = '/api/assets/add'
    payload = {
        'meta': json.dumps(meta),
        'filename': file_path.name,
        'bounding_boxes': json.dumps(bounding_boxes),
        'keypoints': json.dumps(keypoints)
    }
    files = {'file': open(file_path, 'rb')}
    response = self.post(url, payload, files=files)
    print(response.text)

    return Asset.from_json(response.json())

@patch
def delete_assets(self:Box21Api, asset_ids: [int]):
    url = '/api/assets/delete'
    payload = {
        'asset_ids': json.dumps(asset_ids)
    }
    response = self.post(url, payload)
