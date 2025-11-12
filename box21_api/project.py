"""Box21 project"""

# %% auto 0
__all__ = ['Project']

from fastcore.utils import * # type: ignore

from typing import Any, Dict, Optional
from datetime import datetime
import email.utils


class Project:
    "Class that represents a Box21 Project"
    def __init__(self, 
                 id: int,
                 name: str,
                 description: Optional[str] = None,
                 size: int = 0,
                 created_at: Optional[datetime] = None,
                 updated_at: Optional[datetime] = None,
                 deleted_at: Optional[datetime] = None,
                 owner_user_id: Optional[int] = None):
        self.id = id
        self.name = name
        self.description = description
        self.size = size
        self.created_at = created_at
        self.updated_at = updated_at
        self.deleted_at = deleted_at
        self.owner_user_id = owner_user_id

    def __repr__(self):
        return f"Project({self.name}, id={self.id})"
    
    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> 'Project':
        """Create a Project instance from JSON data"""
        def parse_datetime(date_str):
            """Parse datetime string in various formats"""
            if not date_str:
                return None
            if isinstance(date_str, datetime):
                return date_str
            
            try:
                # Try ISO format first
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try RFC 2822 format (e.g., 'Thu, 28 Jul 2022 12:10:17 GMT')
                    timestamp = email.utils.parsedate_to_datetime(date_str)
                    return timestamp
                except (ValueError, TypeError):
                    try:
                        # Try other common formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
                    except:
                        pass
                    
                    # If all parsing fails, return None
                    print(f"Warning: Could not parse datetime string: {date_str}")
                    return None

        # Parse datetime strings if they exist
        created_at = parse_datetime(json_dict.get('created_at'))
        updated_at = parse_datetime(json_dict.get('updated_at'))
        deleted_at = parse_datetime(json_dict.get('deleted_at'))

        project = cls(
            id=json_dict['id'],
            name=json_dict['name'],
            description=json_dict.get('description'),
            size=json_dict.get('size', 0),
            created_at=created_at,
            updated_at=updated_at,
            deleted_at=deleted_at,
            owner_user_id=json_dict.get('owner_user_id')
        )
        return project 