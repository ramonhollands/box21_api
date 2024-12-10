"""Box21 asset"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../02_asset.ipynb.

# %% auto 0
__all__ = ['Asset']

# %% ../02_asset.ipynb 2
from fastcore.utils import * # type: ignore

# %% ../02_asset.ipynb 4
from typing import Any, Dict, List

from .annotation import Box21Annotation, parse_json_annotation


class Asset:
    def __init__(self, 
                 deleted: str, # deleted
                 id: int, # id
                 in_validation_set: bool, # whether the asset is in the validation set
                 liked: bool, # whether the asset is liked in Box21
                 meta: str, # json dict with key value pairs
                 original_category: str, # optional category
                 path: str, # box21 asset path
                 project_id: int, # box21 asset project_id
                 unclear:bool, # whether the asset is marked as unclear
                 validated: bool, # whether the asset is marked as validated
                 annotations: Optional[List[Box21Annotation]] = None
                ):
        "Create a new asset, including the following parameters."
        self.deleted = deleted
        self.id = id
        self.in_validation_set = in_validation_set
        self.liked = liked
        self.meta = meta
        self.original_category = original_category
        self.path = path
        self.project_id = project_id
        self.unclear = unclear
        self.validated = validated
        self.annotations = annotations
        
    def __repr__(self):
        return f"Asset({self.meta})"
    
    @classmethod
    def from_json(cls, json_dict : Dict[str, Any]) -> 'Asset':

        annotations : List[Box21Annotation] = []
        if 'relationships' in json_dict:
            if 'annotations' in json_dict['relationships']:
                for annotation_json in json_dict['relationships']['annotations']:
                    annotations.append(parse_json_annotation(annotation_json))

        asset = cls(
            json_dict['deleted'],
            json_dict['id'],
            json_dict['in_validation_set'],
            json_dict['liked'],
            json_dict['meta'],
            json_dict['original_category'],
            json_dict['path'],
            json_dict['project_id'],
            json_dict['unclear'],
            json_dict['validated'],
            annotations
        )
        return asset
    "Draw `n` cards."
