#!/usr/bin/env python3
"""
Box21 Import Tool

A command-line tool to import Box21 projects including assets, annotations, and labels.
"""

import os
import json
import getpass
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys
from PIL import Image
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import Box21Api
from .asset import Asset
from .annotation import Box21Annotation
from .label import Box21Label


def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def get_password(prompt: str = "Password") -> str:
    """Get password input without echoing."""
    return getpass.getpass(f"{prompt}: ")


def select_from_list(items: List[Any], display_key: str, id_key: str = "id") -> Any:
    """Display a list of items and let user select one."""
    print(f"\nAvailable {display_key}s:")
    for i, item in enumerate(items, 1):
        # Handle both dictionary and object access
        if hasattr(item, display_key):
            name = getattr(item, display_key, 'Unknown')
        else:
            name = item.get(display_key, 'Unknown') if hasattr(item, 'get') else 'Unknown'
            
        if hasattr(item, id_key):
            item_id = getattr(item, id_key, 'Unknown')
        else:
            item_id = item.get(id_key, 'Unknown') if hasattr(item, 'get') else 'Unknown'
            
        # Try to get label_count if available
        label_count = 0
        if hasattr(item, 'label_count'):
            label_count = getattr(item, 'label_count', 0)
        elif hasattr(item, 'get'):
            label_count = item.get('label_count', 0)
            
        print(f"{i}. {name} (ID: {item_id}, Labels: {label_count})")
    
    while True:
        try:
            choice = int(input(f"\nSelect {display_key} (1-{len(items)}): ")) - 1
            if 0 <= choice < len(items):
                return items[choice]
            else:
                print(f"Please enter a number between 1 and {len(items)}")
        except ValueError:
            print("Please enter a valid number")


def select_import_directory() -> Path:
    """Let user select an import directory."""
    while True:
        import_dir = get_user_input("Enter the path to the export directory to import from")
        import_path = Path(import_dir)
        
        if not import_path.exists():
            print(f"Directory does not exist: {import_path}")
            continue
            
        if not import_path.is_dir():
            print(f"Path is not a directory: {import_path}")
            continue
            
        # Check if it looks like a valid export directory
        assets_dir = import_path / 'assets'
        annotations_dir = import_path / 'annotations'
        
        if not assets_dir.exists() or not annotations_dir.exists():
            print(f"Directory does not appear to be a valid Box21 export (missing assets/ or annotations/ subdirectories)")
            continue
            
        return import_path


def get_asset_files(import_dir: Path) -> List[Path]:
    """Get all asset image files from the import directory."""
    assets_dir = import_dir / 'assets'
    if not assets_dir.exists():
        return []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    asset_files = []
    
    for file_path in assets_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            asset_files.append(file_path)
    
    return sorted(asset_files, key=lambda x: int(x.stem) if x.stem.isdigit() else 0)


def get_annotation_files(import_dir: Path) -> List[Path]:
    """Get all annotation files from the import directory."""
    annotations_dir = import_dir / 'annotations'
    if not annotations_dir.exists():
        return []
    
    annotation_files = []
    for file_path in annotations_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.json':
            annotation_files.append(file_path)
    
    return sorted(annotation_files, key=lambda x: int(x.stem) if x.stem.isdigit() else 0)


def load_assets_metadata(import_dir: Path) -> Dict[str, Any]:
    """Load assets metadata from assets.json file."""
    assets_file = import_dir / 'assets.json'
    if not assets_file.exists():
        print("Warning: assets.json not found, using default validation settings")
        return {}
    
    try:
        with open(assets_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load assets.json: {e}")
        return {}


def import_single_asset(api: Box21Api, asset_file: Path, annotation_file: Path, label_name2id: Dict[str, int], asset_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Import a single asset with its annotations."""
    asset_id = int(asset_file.stem)
    
    try:
        # Get asset metadata
        asset_info = asset_metadata.get(str(asset_id), {})
        validated = asset_info.get('validated', False)
        in_validation_set = asset_info.get('in_validation_set', False)
        
        # Prepare metadata (empty for now)
        meta = {}
        
        # Load annotations and convert to Annotation objects
        annotations = []
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            for annotation in annotation_data:
                label_name = annotation.get('label_name', 'Unknown')

                certainty = annotation.get('certainty', 1.0)
                
                if annotation.get('type') == 'bounding_box':
                    # Create BoundingBox annotation
                    from .annotation import BoundingBox
                    bbox = BoundingBox(
                        label_name=label_name,
                        certainty=certainty,
                        x=annotation['x'],
                        y=annotation['y'],
                        width=annotation['width'],
                        height=annotation['height']
                    )
                    annotations.append(bbox)
                elif annotation.get('type') == 'keypoint':
                    # Create Keypoint annotation
                    from .annotation import Keypoint
                    kp = Keypoint(
                        label_name=label_name,
                        certainty=certainty,
                        x=annotation['x'],
                        y=annotation['y']
                    )
                    annotations.append(kp)
        
        # Upload asset with annotations
        uploaded_assets = api.add_asset(
            file_path=asset_file,
            meta=meta,
            annotations=annotations,
            validated=validated,
            in_validation_set=in_validation_set,
            no_duplicate_filename=False
        )
        
        # The API returns a list of assets, but we're uploading one
        uploaded_asset = uploaded_assets[0] if uploaded_assets else None
        
        if uploaded_asset is None:
            return {
                'original_id': asset_id,
                'new_asset_id': None,
                'annotations_added': 0,
                'status': 'error',
                'error': 'Failed to upload asset'
            }
        
        return {
            'original_id': asset_id,
            'new_asset_id': uploaded_asset.id,
            'annotations_added': len(annotations),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'original_id': asset_id,
            'new_asset_id': None,
            'annotations_added': 0,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main function for the box21_import command."""
    parser = argparse.ArgumentParser(description='Import Box21 project data')
    parser.add_argument('--host', help='Box21 host URL')
    parser.add_argument('--username', help='Box21 username/email')
    parser.add_argument('--import-dir', help='Import directory path')
    parser.add_argument('--project-id', type=int, help='Target Box21 project ID')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for parallel processing (default: 10)')
    
    args = parser.parse_args()
    
    print("Box21 Import Tool")
    print("=" * 50)
    
    # Get connection details
    host = args.host or get_user_input("Enter Box21 host", "https://box21.ai")
    username = args.username or get_user_input("Enter username/email")
    password = get_password("Enter password")
    
    # Extract port from host if specified
    if ':' in host and not host.startswith('http'):
        host_parts = host.split(':')
        if len(host_parts) == 2:
            host = host_parts[0]
            port = int(host_parts[1])
        else:
            port = 443
    else:
        if host.startswith('https'):
            port = 443
        else:
            port = 80
    
    # Clean up host URL
    if not host.startswith(('http://', 'https://')):
        host = f"https://{host}"
    
    print(f"\nConnecting to {host}:{port}...")
    
    try:
        # Create API instance
        api = Box21Api(email=username, password=password, host=host, port=port, project_id=0)
        
        # Get target project
        project_id = args.project_id
        if project_id is None:
            print("Fetching available projects...")
            projects = api.get_projects()
            
            if not projects:
                print("No projects found or unable to retrieve projects.")
                print("Please provide a project ID manually.")
                while True:
                    try:
                        project_id_input = get_user_input("Enter target Box21 project ID")
                        project_id = int(project_id_input)
                        break
                    except ValueError:
                        print("Please enter a valid number for the project ID")
            else:
                selected_project = select_from_list(projects, 'name')
                project_id = selected_project.id
                print(f"\nSelected target project: {selected_project.name} (ID: {project_id})")
        else:
            selected_project = {'id': project_id, 'name': f'Project {project_id}'}
        
        # Update API with target project_id
        api.project_id = project_id
        
        # Get import directory
        import_dir = args.import_dir
        if import_dir is None:
            import_dir = select_import_directory()
        else:
            import_dir = Path(import_dir)
            if not import_dir.exists():
                print(f"Import directory does not exist: {import_dir}")
                return 1
        
        print(f"\nImporting from: {import_dir}")
        print(f"Target project: {selected_project.name} (ID: {project_id})")
        
        # Get asset and annotation files
        asset_files = get_asset_files(import_dir)
        annotation_files = get_annotation_files(import_dir)
        
        if not asset_files:
            print("No asset files found in the import directory")
            return 1
        
        print(f"Found {len(asset_files)} asset files and {len(annotation_files)} annotation files")
        
        # Load assets metadata
        assets_metadata = load_assets_metadata(import_dir)
        if assets_metadata:
            print(f"Loaded metadata for {len(assets_metadata)} assets")
        
        # Get labels from target project for mapping
        print("Fetching target project labels...")
        labels = api.get_labels()
        label_name2id = {label.name: label.id for label in labels}
        print(f"Found {len(label_name2id)} labels in target project")
        
        # Create annotation file mapping
        annotation_file_map = {int(f.stem): f for f in annotation_files}
        
        # Import assets in parallel
        print(f"\nStarting import with batch size {args.batch_size}...")
        
        results = []
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            # Submit all import tasks
            future_to_asset = {}
            for asset_file in asset_files:
                asset_id = int(asset_file.stem)
                annotation_file = annotation_file_map.get(asset_id, Path())
                future = executor.submit(import_single_asset, api, asset_file, annotation_file, label_name2id, assets_metadata)
                future_to_asset[future] = asset_file
            
            # Process completed imports with progress bar
            with tqdm(total=len(asset_files), desc="Importing assets", unit="asset") as pbar:
                for future in as_completed(future_to_asset):
                    asset_file = future_to_asset[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                        pbar.set_postfix({'Asset ID': asset_file.stem})
                    except Exception as e:
                        print(f"Error processing asset {asset_file.stem}: {e}")
                        pbar.update(1)
        
        # Print summary
        successful_imports = [r for r in results if r['status'] == 'success']
        failed_imports = [r for r in results if r['status'] == 'error']
        total_annotations = sum(r['annotations_added'] for r in successful_imports)
        
        print(f"\nImport completed!")
        print(f"Successfully imported: {len(successful_imports)} assets")
        print(f"Failed imports: {len(failed_imports)} assets")
        print(f"Total annotations added: {total_annotations}")
        
        if failed_imports:
            print("\nFailed imports:")
            for failed in failed_imports:
                print(f"  Asset {failed['original_id']}: {failed.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Login failed, correct username and password?")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 