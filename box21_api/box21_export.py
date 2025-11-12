#!/usr/bin/env python3
"""
Box21 Export Tool

A command-line tool to export Box21 projects including assets, annotations, and labels.
"""

import os
import json
import getpass
import argparse
import requests
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


def export_labels(api: Box21Api, export_dir: Path) -> List[Box21Label]:
    """Export all labels for the project."""
    print("Exporting labels...")
    labels = api.get_labels()
    
    labels_data = []
    for label in labels:
        label_data = {
            'id': label.id,
            'name': label.name,
            'parent_id': label.parent_id,
            'project_id': label.project_id,
            'type': label.type,
            'annotation_stats': [
                {
                    'annotation_type': stat.annotation_type,
                    'id': stat.id,
                    'label_id': stat.label_id,
                    'processed_count': stat.processed_count,
                    'to_be_processed_count': stat.to_be_processed_count
                }
                for stat in label.annotation_stats
            ]
        }
        labels_data.append(label_data)
    
    labels_file = export_dir / 'labels.json'
    with open(labels_file, 'w') as f:
        json.dump(labels_data, f, indent=2)
    
    print(f"Exported {len(labels)} labels to {labels_file}")
    return labels


def download_single_asset(api: Box21Api, asset: Asset, assets_dir: Path, annotations_dir: Path, label_id2name: Dict[int, str]) -> Dict[str, Any]:
    """Download a single asset image and return asset data."""
    image_filename = f"{asset.id}.jpg"
    image_path = assets_dir / image_filename
    
    try:
        # Download the asset image using the API method
        img = api.download_asset(asset.id)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPG with good quality
        img.save(image_path, 'JPEG', quality=95, optimize=True)
        
        # Process annotations that are already included in the asset
        annotations_data = []
        if hasattr(asset, 'annotations') and asset.annotations:
            for annotation in asset.annotations:
                if hasattr(annotation, 'x') and hasattr(annotation, 'y'):
                    if hasattr(annotation, 'width') and hasattr(annotation, 'height'):
                        # Bounding box
                        annotation_data = {
                            'id': annotation.id,
                            'asset_id': annotation.asset_id,
                            'label_id': annotation.label_id,
                            'label_name': label_id2name.get(annotation.label_id, 'Unknown'),
                            'certainty': annotation.certainty,
                            'validated': annotation.validated,
                            'type': 'bounding_box',
                            'x': annotation.x,
                            'y': annotation.y,
                            'width': annotation.width,
                            'height': annotation.height
                        }
                    else:
                        # Keypoint
                        annotation_data = {
                            'id': annotation.id,
                            'asset_id': annotation.asset_id,
                            'label_id': annotation.label_id,
                            'label_name': label_id2name.get(annotation.label_id, 'Unknown'),
                            'certainty': annotation.certainty,
                            'validated': annotation.validated,
                            'type': 'keypoint',
                            'x': annotation.x,
                            'y': annotation.y
                        }
                    annotations_data.append(annotation_data)
        
        # Save annotations to separate file
        annotation_file = annotations_dir / f"{asset.id}.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations_data, f, indent=2)
        
        asset_data = {
            'id': asset.id,
            'deleted': asset.deleted,
            'in_validation_set': asset.in_validation_set,
            'liked': asset.liked,
            'meta': asset.meta,
            'original_category': asset.original_category,
            'path': asset.path,
            'project_id': asset.project_id,
            'unclear': asset.unclear,
            'validated': asset.validated,
            'local_image_path': str(image_path) if image_path.exists() else None,
            'annotations': annotations_data
        }
        
        return asset_data
        
    except Exception as e:
        print(f"Warning: Could not download/convert image for asset {asset.id}: {e}")
        # Return asset data without image if download fails
        asset_data = {
            'id': asset.id,
            'deleted': asset.deleted,
            'in_validation_set': asset.in_validation_set,
            'liked': asset.liked,
            'meta': asset.meta,
            'original_category': asset.original_category,
            'path': asset.path,
            'project_id': asset.project_id,
            'unclear': asset.unclear,
            'validated': asset.validated,
            'local_image_path': None,
            'annotations': []
        }
        
        # Save empty annotations file
        annotation_file = annotations_dir / f"{asset.id}.json"
        with open(annotation_file, 'w') as f:
            json.dump([], f, indent=2)
        
        return asset_data


def export_assets_batch(api: Box21Api, export_dir: Path, labels: List[Box21Label], batch_size: int = 1000, max_workers: int = 50) -> List[Asset]:
    """Export all assets in batches using the v2 endpoint with parallel downloading."""
    print("Exporting assets...")
    
    all_assets = []
    offset = 0
    total_exported = 0
    
    # Create assets directory for images
    assets_dir = export_dir / 'assets'
    assets_dir.mkdir(exist_ok=True)
    print(f"Created assets directory: {assets_dir}")
    
    # Create annotations directory for individual annotation files
    annotations_dir = export_dir / 'annotations'
    annotations_dir.mkdir(exist_ok=True)
    print(f"Created annotations directory: {annotations_dir}")
    
    # Create label ID to name mapping
    label_id2name = {label.id: label.name for label in labels}
    print(f"Created label mapping for {len(label_id2name)} labels")
    
    # Initialize assets.json file
    assets_file = export_dir / 'assets.json'
    assets_data = {}
    
    while True:
        print(f"Fetching assets batch {offset//batch_size + 1}...")
        assets_batch = api.get_assets(offset=offset, limit=batch_size)
        
        if not assets_batch:
            break
            
        # Process this batch in parallel with progress bar
        print(f"Downloading {len(assets_batch)} assets in parallel...")
        
        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_asset = {
                executor.submit(download_single_asset, api, asset, assets_dir, annotations_dir, label_id2name): asset 
                for asset in assets_batch
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(assets_batch), desc="Downloading assets", unit="asset") as pbar:
                for future in as_completed(future_to_asset):
                    asset = future_to_asset[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        pbar.update(1)
                        pbar.set_postfix({'Asset ID': asset.id})
                    except Exception as e:
                        print(f"Error processing asset {asset.id}: {e}")
                        pbar.update(1)
        
        # Add asset data to assets.json
        for result in batch_results:
            if result and 'id' in result:
                assets_data[result['id']] = {
                    'validated': result.get('validated', False),
                    'in_validation_set': result.get('in_validation_set', False),
                    'local_image_path': result.get('local_image_path'),
                    'original_id': result['id']
                }
        
        all_assets.extend(assets_batch)
        total_exported += len(assets_batch)
        print(f"Exported {total_exported} assets so far...")
        
        if len(assets_batch) < batch_size:
            break
            
        offset += batch_size
    
    # Save assets.json
    with open(assets_file, 'w') as f:
        json.dump(assets_data, f, indent=2)
    
    print(f"Exported {len(all_assets)} assets")
    print(f"Assets metadata saved to {assets_file}")
    return all_assets


def create_export_summary(export_dir: Path, project_info: Dict[str, Any], 
                         labels: List[Box21Label], assets: List[Asset]):
    """Create a summary file with export information."""
    summary = {
        'export_info': {
            'project_id': project_info.get('id'),
            'project_name': project_info.get('name', 'Unknown'),
            'export_timestamp': str(Path().cwd().stat().st_mtime),
            'export_directory': str(export_dir)
        },
        'statistics': {
            'total_labels': len(labels),
            'total_assets': len(assets),
            'total_annotations': sum(len(asset.annotations or []) for asset in assets),
            'validated_assets': len([a for a in assets if a.validated]),
            'assets_in_validation_set': len([a for a in assets if a.in_validation_set])
        }
    }
    
    summary_file = export_dir / 'export_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Export summary created: {summary_file}")


def main():
    """Main function for the box21_export command."""
    parser = argparse.ArgumentParser(description='Export Box21 project data')
    parser.add_argument('project_id', type=int, nargs='?', help='Box21 project ID to export (optional, will prompt if not provided)')
    parser.add_argument('--host', help='Box21 host URL')
    parser.add_argument('--username', help='Box21 username/email')
    parser.add_argument('--batch-size', type=int, default=500, 
                       help='Batch size for asset retrieval (default: 500)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Maximum number of parallel downloads (default: 10)')
    parser.add_argument('--export-dir', help='Export directory (default: ~/.box21_exports/project_id/)')
    
    args = parser.parse_args()
    
    print("Box21 Export Tool")
    print("=" * 50)
    
    # Get connection details first
    host = args.host or get_user_input("Enter Box21 host", "https://box21.ai")
    username = args.username or get_user_input("Enter username/email")
    password = get_password("Enter password")
    
    # Extract port from host if specified
    if ':' in host and not host.startswith('http'):
        # Handle case where port is in host string
        host_parts = host.split(':')
        if len(host_parts) == 2:
            host = host_parts[0]
            port = int(host_parts[1])
        else:
            port = 443
    else:
        # Default port based on protocol
        if host.startswith('https'):
            port = 443
        else:
            port = 80
    
    # Clean up host URL
    if not host.startswith(('http://', 'https://')):
        host = f"https://{host}"
    
    print(f"\nConnecting to {host}:{port}...")
    
    try:
        # Create API instance without project_id first
        api = Box21Api(email=username, password=password, host=host, port=port, project_id=0)
        
        # Get project_id if not provided
        project_id = args.project_id
        if project_id is None:
            print("Fetching available projects...")
            projects = api.get_projects()
            
            if not projects:
                print("No projects found or unable to retrieve projects.")
                print("Please provide a project ID manually.")
                while True:
                    try:
                        project_id_input = get_user_input("Enter Box21 project ID")
                        project_id = int(project_id_input)
                        break
                    except ValueError:
                        print("Please enter a valid number for the project ID")
            else:
                # Let user select project
                selected_project = select_from_list(projects, 'name')
                project_id = selected_project.id
                print(f"\nSelected project: {selected_project.name} (ID: {project_id})")
        else:
            # Use provided project_id
            selected_project = {'id': project_id, 'name': f'Project {project_id}'}
        
        # Update API with correct project_id
        api.project_id = project_id
        
        print(f"\nExporting project: {selected_project.name} (ID: {project_id})")
        
        # Create export directory
        if args.export_dir:
            export_dir = Path(args.export_dir)
        else:
            export_dir = Path.home() / '.box21_exports' / str(project_id)
        
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting to: {export_dir}")
        
        # Export data
        labels = export_labels(api, export_dir)
        assets = export_assets_batch(api, export_dir, labels, args.batch_size, args.max_workers)
        
        # Create summary
        project_info = {
            'id': selected_project.id,
            'name': selected_project.name
        }
        create_export_summary(export_dir, project_info, labels, assets)
        
        print(f"\nExport completed successfully!")
        print(f"Total labels: {len(labels)}")
        print(f"Total assets: {len(assets)}")
        print(f"Export location: {export_dir}")
        
    except Exception as e:
        print(f"Login failed, correct username and password?")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 