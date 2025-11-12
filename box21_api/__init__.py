__version__ = "0.0.18"

# Import main classes for easy access
from .box21_api import Box21Api
from .asset import Asset
from .annotation import Box21Annotation, Box21BoundingBox, Box21Keypoint, Annotation, BoundingBox, Keypoint
from .label import Box21Label
from .annotation_stat import AnnotationStat
from .job import Job
from .project import Project
from .export import export_asset, export_assets

# Make these available at package level
__all__ = [
    'Box21Api',
    'Asset', 
    'Box21Annotation',
    'Box21BoundingBox',
    'Box21Keypoint',
    'Annotation',
    'BoundingBox',
    'Keypoint',
    'Box21Label',
    'AnnotationStat',
    'Job',
    'Project',
    'export_asset',
    'export_assets'
]
