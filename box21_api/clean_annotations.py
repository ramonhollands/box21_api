#!/usr/bin/env python3
"""
Box21 Clean Annotations Tool

A command-line tool to remove annotations that are outside image boundaries.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import Box21Api
from .asset import Asset
from .annotation import Box21Annotation, Box21BoundingBox, Box21Keypoint


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
            
        # Try to get additional info if available
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


def is_annotation_outside_image(annotation: Box21Annotation, image_width: int, image_height: int) -> bool:
    """Check if an annotation is outside the image boundaries."""
    if isinstance(annotation, Box21BoundingBox):
        # For bounding boxes, check if any part is outside
        x, y, width, height = annotation.x, annotation.y, annotation.width, annotation.height
        
        # Convert normalized coordinates to pixel coordinates
        x_px = x * image_width
        y_px = y * image_height
        width_px = width * image_width
        height_px = height * image_height
        
        # Check if box is completely outside image
        if (x_px + width_px <= 0 or  # Box is to the left
            y_px + height_px <= 0 or  # Box is above
            x_px >= image_width or   # Box is to the right
            y_px >= image_height):   # Box is below
            return True
        
        # Check if box is too small (less than 1 pixel)
        if width_px < 1 or height_px < 1:
            return True
            
    elif isinstance(annotation, Box21Keypoint):
        # For keypoints, check if point is outside
        x, y = annotation.x, annotation.y
        
        # Convert normalized coordinates to pixel coordinates
        x_px = x * image_width
        y_px = y * image_height
        
        # Check if point is outside image
        if (x_px < 0 or x_px >= image_width or 
            y_px < 0 or y_px >= image_height):
            return True
    
    return False


def clean_asset_annotations(api: Box21Api, asset: Asset) -> Dict[str, Any]:
    """Clean annotations for a single asset."""
    try:
        # Download the asset image to get dimensions
        img = api.download_asset(asset.id)
        image_width, image_height = img.size
        
        # Get annotations for this asset
        annotations = api.get_annotations(asset.id)
        
        if not annotations:
            return {
                'asset_id': asset.id,
                'total_annotations': 0,
                'removed_annotations': 0,
                'status': 'no_annotations'
            }
        
        # Find annotations outside image boundaries
        annotations_to_remove = []
        for annotation in annotations:
            if is_annotation_outside_image(annotation, image_width, image_height):
                annotations_to_remove.append(annotation)
        
        # Remove invalid annotations
        if annotations_to_remove:
            api.detete_annotations(annotations_to_remove)
        
        return {
            'asset_id': asset.id,
            'total_annotations': len(annotations),
            'removed_annotations': len(annotations_to_remove),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'asset_id': asset.id,
            'total_annotations': 0,
            'removed_annotations': 0,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main function for the clean_annotations command."""
    parser = argparse.ArgumentParser(description='Clean annotations outside image boundaries')
    parser.add_argument('--host', help='Box21 host URL')
    parser.add_argument('--username', help='Box21 username/email')
    parser.add_argument('--project-id', type=int, help='Box21 project ID')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for parallel processing (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be removed without actually removing')
    
    args = parser.parse_args()
    
    print("Box21 Clean Annotations Tool")
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
        
        print(f"\nCleaning annotations for project: {selected_project.name} (ID: {project_id})")
        
        if args.dry_run:
            print("DRY RUN MODE: No annotations will be actually removed")
        
        # Get all assets
        print("Fetching assets...")
        all_assets = []
        offset = 0
        batch_size = 1000
        
        while True:
            assets_batch = api.get_assets(offset=offset, limit=batch_size)
            if not assets_batch:
                break
            all_assets.extend(assets_batch)
            offset += batch_size
            if len(assets_batch) < batch_size:
                break
        
        print(f"Found {len(all_assets)} assets to process")
        
        if not all_assets:
            print("No assets found in this project")
            return 0
        
        # Confirm before proceeding
        if not args.dry_run:
            confirm = input(f"\nThis will permanently remove invalid annotations. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Operation cancelled")
                return 0
        
        # Clean annotations in parallel
        print(f"\nStarting annotation cleaning with batch size {args.batch_size}...")
        
        results = []
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            # Submit all cleaning tasks
            future_to_asset = {
                executor.submit(clean_asset_annotations, api, asset): asset 
                for asset in all_assets
            }
            
            # Process completed cleanings with progress bar
            with tqdm(total=len(all_assets), desc="Cleaning annotations", unit="asset") as pbar:
                for future in as_completed(future_to_asset):
                    asset = future_to_asset[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                        pbar.set_postfix({'Asset ID': asset.id})
                    except Exception as e:
                        print(f"Error processing asset {asset.id}: {e}")
                        pbar.update(1)
        
        # Print summary
        successful_cleanings = [r for r in results if r['status'] == 'success']
        error_cleanings = [r for r in results if r['status'] == 'error']
        no_annotations = [r for r in results if r['status'] == 'no_annotations']
        
        total_annotations = sum(r['total_annotations'] for r in successful_cleanings)
        total_removed = sum(r['removed_annotations'] for r in successful_cleanings)
        
        print(f"\nCleaning completed!")
        print(f"Successfully processed: {len(successful_cleanings)} assets")
        print(f"Assets with errors: {len(error_cleanings)} assets")
        print(f"Assets with no annotations: {len(no_annotations)} assets")
        print(f"Total annotations processed: {total_annotations}")
        print(f"Total annotations removed: {total_removed}")
        
        if args.dry_run:
            print(f"\nDRY RUN: No annotations were actually removed")
        else:
            print(f"\nSuccessfully removed {total_removed} invalid annotations")
        
        if error_cleanings:
            print("\nAssets with errors:")
            for error in error_cleanings:
                print(f"  Asset {error['asset_id']}: {error.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Login failed, correct username and password?")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 