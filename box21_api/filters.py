"""Filters for your data"""

# %% auto 0
__all__ = ['get_contains_meta_filter', 'get_not_contains_meta_filter', 'get_asset_is_validated_filter',
           'get_asset_is_not_validated_filter', 'get_in_validation_set_filter', 'get_not_in_validation_set_filter',
           'get_number_annotations_filter', 'get_not_containing_annotation_with_name_filter',
           'get_has_suggestions_filter', 'get_has_x_suggestions_filter', 'get_asset_id', 'get_containing_annotation',
           'get_not_containing_annotation', 'get_min_nr_annotations', 'get_added_after_filter',
           'get_min_annotations_confidence_filter', 'get_max_annotations_confidence_filter']

from typing import Any, Dict

def get_contains_meta_filter(contains: str) -> Dict[str, Any]:
    return {"type":7,"value":contains}

def get_not_contains_meta_filter(contains: str) -> Dict[str, Any]:
    return {"type":8,"value":contains}

def get_asset_is_validated_filter() -> Dict[str, Any]:
    return {"type": 9, "value": 'Yes'}

def get_asset_is_not_validated_filter() -> Dict[str, Any]:
    return {"type": 9, "value": 'No'}

def get_in_validation_set_filter() -> Dict[str, Any]:
    return {"type": 13, "value": 'Yes'}

def get_not_in_validation_set_filter() -> Dict[str, Any]:
    return {"type": 13, "value": 'No'}

def get_number_annotations_filter(x: int) -> Dict[str, Any]:
    return {"type": 22, "value": x}

def get_not_containing_annotation_with_name_filter(name: str) -> Dict[str, Any]:
    return {"type": 28, "value": name}

def get_has_suggestions_filter() -> Dict[str, Any]:
    return {"type": 31, "value": 'Yes'}

def get_has_x_suggestions_filter(x: int) -> Dict[str, Any]:
    return {"type": 45, "value": x}

def get_asset_id(x: int) -> Dict[str, Any]:
    return {"type": 36, "value": x}

def get_containing_annotation(name: str) -> Dict[str, Any]:
    return {"type": 1, "value": name}

def get_not_containing_annotation(name: str) -> Dict[str, Any]:
    return {"type": 2, "value": name}

def get_min_nr_annotations(x: int) -> Dict[str, Any]:
    return {"type": 32, "value": x}

def get_added_after_filter(x: str) -> Dict[str, Any]:
    return {"type": 38, "value": x}

def get_added_before_filter(x: str) -> Dict[str, Any]:
    return {"type": 37, "value": x}

def get_min_annotations_confidence_filter(x: float) -> Dict[str, Any]:
    return {"type": 3, "value": x}

def get_max_annotations_confidence_filter(x: float) -> Dict[str, Any]:
    return {"type": 4, "value": x}


