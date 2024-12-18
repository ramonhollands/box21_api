"""Box21 annotation"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_annotation.ipynb.

# %% auto 0
__all__ = ['Annotation', 'Box21Annotation', 'Keypoint', 'Box21Keypoint', 'BoundingBox', 'Box21BoundingBox',
           'parse_json_annotation']

# %% ../03_annotation.ipynb 4
from abc import ABC, abstractmethod, abstractproperty

class Annotation(ABC):
    "Abstract class that represents an Annotation"
    def __init__(self,
                 label_name: str, # label name
                 certainty: float, # certainty of annotation
                ):
        self.label_name = label_name
        self.certainty = certainty

class Box21Annotation(ABC):
    "Abstract class that represents a Box21 Annotation"
    def __init__(self, 
                 asset_id: int, # related asset id in Box21
                 id: int, # unique id in Box21
                 certainty: float, # certainty of annotation
                 label_id: int, # related label id in Box21
                 project_id: int, # related project id in Box21
                 validated: bool # whether the annotation is validated
                ): 
        self.asset_id = asset_id
        self.id = id
        self.certainty = certainty
        self.label_id = label_id
        self.project_id = project_id
        self.validated = validated

# %% ../03_annotation.ipynb 7
class Keypoint(Annotation):
    "Represents a Keypoint Annotation"
    def __init__(self,
                 label_name: str,
                 certainty: float,
                 x: float,
                 y: float
                ):
        super().__init__(label_name, certainty)
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"Keypoint({self.label_name}, {self.x, self.y})"
        

# %% ../03_annotation.ipynb 8
class Box21Keypoint(Box21Annotation):
    "Represents a Box21 Keypoint Annotation"
    def __init__(self,
                 asset_id: int, 
                 id: int,
                 certainty: float,
                 label_id: int,
                 project_id: int,
                 validated: bool,
                 x: int,
                 y: int
                ):
        super().__init__(asset_id, id, certainty, label_id, project_id, validated)
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"Keypoint({self.label_id}, {self.x, self.y})"
        

# %% ../03_annotation.ipynb 10
class BoundingBox(Annotation):
    "Represents a BoundingBox Annotation"
    def __init__(self,
                 label_name: str,
                 certainty: float,
                 x: float,
                 y: float,
                 width: float,
                 height: float
                ):
        super().__init__(label_name, certainty)
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        
    def __repr__(self):
        return f"BoundingBox({self.label_name}, {self.x, self.y, self.width, self.height})"


class Box21BoundingBox(Box21Annotation):
    "Represents a Box21 BoundingBox Annotation"
    def __init__(self,
                 asset_id: int, 
                 id: int,
                 certainty: float,
                 label_id: int,
                 project_id: int,
                 validated: bool,
                 x: float,
                 y: float,
                 width: float,
                 height: float
                ):
        super().__init__(asset_id, id, certainty, label_id, project_id, validated)
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        
    def __repr__(self):

        return f"BoundingBox({self.label_id}, {self.x, self.y, self.width, self.height})"

# %% ../03_annotation.ipynb 11
from typing import Any, Dict
import json

def parse_json_annotation(annotation_json : Dict[str,Any]) -> Box21Annotation:
    asset_id = annotation_json['asset_id']
    annotation_id = annotation_json['id']
    certainty = annotation_json['certainty']
    label_id = annotation_json['label_id']
    project_id = annotation_json['project_id']
    validated = annotation_json['validated']
    coords = json.loads(annotation_json['coords'])

    if annotation_json['type'] == 1:
        x, y, w, h = coords
        return Box21BoundingBox(asset_id, annotation_id, certainty, label_id, project_id, validated, x, y, w, h)
    else:
        x, y = coords
        return Box21Keypoint(asset_id, annotation_id, certainty, label_id, project_id, validated, x, y)
