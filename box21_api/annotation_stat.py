# AUTOGENERATED! DO NOT EDIT! File to edit: ../07_annotation_stat.ipynb.

# %% auto 0
__all__ = ['AnnotationStat']

# %% ../07_annotation_stat.ipynb 2
from fastcore.utils import * # type: ignore

# %% ../07_annotation_stat.ipynb 3
from typing import Any, Dict

class AnnotationStat:
    def __init__(self, 
                 annotation_type: int,
                 id: int,
                 label_id: int,
                 processed_count: int,
                 to_be_processed_count: int
                ):
        self.annotation_type = annotation_type
        self.id = id
        self.label_id = label_id
        self.processed_count = processed_count
        self.to_be_processed_count = to_be_processed_count
        
    def __repr__(self):
        return f"AnnotationStat({self.annotation_type}, {self.id}, {self.label_id}, {self.processed_count}, {self.to_be_processed_count})"
    
    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> 'AnnotationStat':
        "Create an AnnotationStat instance from a JSON dictionary."
        return cls(
            json_dict['annotation_type'],
            json_dict['id'],
            json_dict['label_id'],
            json_dict['processed_count'],
            json_dict['to_be_processed_count'],
        )

