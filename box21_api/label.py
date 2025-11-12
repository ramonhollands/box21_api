"""Box21 label"""

# %% auto 0
__all__ = ['Box21Label']

from abc import ABC, abstractmethod, abstractproperty
from typing import List
from .annotation_stat import AnnotationStat

class Box21Label(ABC):
    "Class that represents a Box21 Label"
    def __init__(self, 
                 id: int,
                 name: float,
                 parent_id: int,
                 project_id: int,
                 annotation_stats: List[AnnotationStat],
                 type: int):
        self.id = id
        self.name = name
        self.parent_id = parent_id
        self.project_id = project_id
        self.type = type
        self.annotation_stats = annotation_stats

    def __repr__(self):
        
        type_label = 'Bounding box'
        if self.type == 2:
            type_label = 'Keypoint'
        
        return f"Label({type_label, self.name, self.id})"
