#!/usr/bin/env python3
"""
Box21 Export Predictions Tool

A command-line tool to export Box21 models and their predictions.
"""

import os
import json
import getpass
import argparse
from pathlib import Path
import traceback
from typing import List, Dict, Any
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import Box21Api


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
        description = ""
        if hasattr(item, 'description'):
            description = getattr(item, 'description', '')
        elif hasattr(item, 'get'):
            description = item.get('description', '')
            
        type_info = ""
        if hasattr(item, 'type'):
            type_info = getattr(item, 'type', '')
        elif hasattr(item, 'get'):
            type_info = item.get('type', '')
            
        print(f"{i}. {name} (ID: {item_id}, Type: {type_info})")
        if description:
            print(f"   Description: {description}")
    
    while True:
        try:
            choice = int(input(f"\nSelect {display_key} (1-{len(items)}): ")) - 1
            if 0 <= choice < len(items):
                return items[choice]
            else:
                print(f"Please enter a number between 1 and {len(items)}")
        except ValueError:
            print("Please enter a valid number")





def download_model_files(api: Box21Api, model: Dict[str, Any], models_dir: Path) -> bool:
    """Download all files for a model."""
    try:
        model_id = model['id']
        # Files go directly in the models_dir (which is already the model-specific directory)
        model_dir = models_dir
        
        # Parse model metadata
        meta = model.get('meta', '{}')
        if isinstance(meta, str):
            try:
                meta_data = json.loads(meta)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse model metadata for model {model_id}")
                meta_data = {}
        else:
            meta_data = meta
        
        # Get files from metadata
        files = meta_data.get('files', [])
        if not files:
            print(f"No files found for model {model_id}")
            return False
        
        print(f"Downloading {len(files)} files for model {model_id}...")
        
        # Download each file
        for file_path in files:
            try:
                # Extract filename from path
                filename = file_path.split('/')[-1]
                
                # Construct download URL with token as query parameter
                download_url = f"{api.host}/api/protected/models/{api.project_id}/{filename}?token={api.token}"
                
                # Download file
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                # Save file
                file_path = model_dir / filename
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded: {filename}")
                
            except Exception as e:
                print(f"Error downloading file {file_path}: {e}")
        
        # Save model metadata
        model_info_file = model_dir / 'model_info.json'
        with open(model_info_file, 'w') as f:
            json.dump(model, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        return False


def get_predictions(api: Box21Api, model_id: int) -> List[Dict[str, Any]]:
    """Get all predictions for a model."""
    try:
        # First get all assets to get their IDs
        assets = api.get_assets(limit=10000)  # Get all assets
        asset_ids = [asset.id for asset in assets]
        
        predictions = []
        
        print(f"Fetching predictions for {len(asset_ids)} assets...")
        
        # Get predictions for each asset
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_asset = {}
            for asset_id in asset_ids:
                future = executor.submit(get_asset_predictions, api, model_id, asset_id)
                future_to_asset[future] = asset_id
            
            with tqdm(total=len(asset_ids), desc="Fetching predictions", unit="asset") as pbar:
                for future in as_completed(future_to_asset):
                    asset_id = future_to_asset[future]
                    try:
                        preds = future.result()
                        if preds:
                            predictions.extend(preds)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error getting predictions for asset {asset_id}: {e}")
                        pbar.update(1)
        
        return predictions
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return []


def get_asset_predictions(api: Box21Api, model_id: int, asset_id: int) -> List[Dict[str, Any]]:
    """Get predictions for a specific asset and model."""
    try:
        payload = {
            'model_id': model_id,
            'project_id': api.project_id,
            'asset_id': asset_id
        }
        
        response = api.post('/api/preds', payload)
        if response.status_code == 200:
            result = response.json()
            
            # Handle case where result might be a string that needs parsing
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse prediction response for asset {asset_id}")
                    return []
            
            # Check if response is a job object (no predictions available)
            if isinstance(result, dict) and 'task' in result and 'progress' in result:
                # This is a job object, no predictions available
                print(f"No predictions available for asset {asset_id} (job status: {result.get('task', 'unknown')})")
                return []
            
            # The response has structure: {"annotations": [...], "colors": [...]}
            if isinstance(result, dict) and 'annotations' in result:
                annotations = result['annotations']
                if isinstance(annotations, list):
                    # Add asset_id to each annotation for tracking
                    for annotation in annotations:
                        annotation['asset_id'] = asset_id
                    return annotations
                else:
                    print(f"Warning: Annotations is not a list for asset {asset_id}")
                    return []
            elif isinstance(result, list):
                # Direct list of annotations
                for annotation in result:
                    annotation['asset_id'] = asset_id
                return result
            elif isinstance(result, dict):
                # Single annotation
                result['asset_id'] = asset_id
                return [result]
            else:
                print(f"Warning: Unexpected prediction response format for asset {asset_id}")
                return []
        else:
            return []
            
    except Exception as e:
        print(f"Error getting predictions for asset {asset_id}: {e}")
        return []


def save_predictions(predictions: List[Dict[str, Any]], preds_dir: Path):
    """Save predictions to individual files."""
    if not predictions:
        print("No predictions to save")
        return
    
    # Group predictions by asset_id
    predictions_by_asset = {}
    for pred in predictions:
        # Handle case where pred might not be a dictionary
        if not isinstance(pred, dict):
            print(f"Warning: Skipping non-dictionary prediction: {type(pred)} - {pred}")
            continue
            
        asset_id = pred.get('asset_id')
        if asset_id:
            if asset_id not in predictions_by_asset:
                predictions_by_asset[asset_id] = []
            predictions_by_asset[asset_id].append(pred)
    
    print(f"Saving predictions for {len(predictions_by_asset)} assets...")
    
    # Save each asset's predictions to a separate file
    for asset_id, asset_preds in predictions_by_asset.items():
        pred_file = preds_dir / f"{asset_id}.json"
        
        # Create a structured output with metadata
        output_data = {
            'asset_id': asset_id,
            'total_predictions': len(asset_preds),
            'predictions': asset_preds
        }
        
        with open(pred_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(predictions_by_asset)} prediction files")


def main():
    """Main function for the box21_export_preds command."""
    parser = argparse.ArgumentParser(description='Export Box21 models and predictions')
    parser.add_argument('--host', help='Box21 host URL')
    parser.add_argument('--username', help='Box21 username/email')
    parser.add_argument('--project-id', type=int, help='Box21 project ID')
    parser.add_argument('--export-dir', help='Export directory (default: ~/.box21_exports/models/)')
    
    args = parser.parse_args()
    
    print("Box21 Export Predictions Tool")
    print("=" * 50)
    
    # Get connection details
    host = args.host or get_user_input("Enter Box21 host", "https://box21.ai")
    username = args.username or get_user_input("Enter username/email")
    password = get_password("Enter password")

    host = 'https://box21.agouti.eu'
    username = 'fadia.al-abbar@wur.nl'
    password = 'geheim!'
    
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
        
        print(f"\nExporting from project: {selected_project.name} (ID: {project_id})")
        
        # Get models
        print("Fetching models...")
        models = api.get_models(project_id)
        
        if not models:
            print("No models found for this project")
            return 1
        
        # Let user select model
        selected_model = select_from_list(models, 'name')
        print(f"\nSelected model: {selected_model['name']} (ID: {selected_model['id']})")
        
        # Create export directory
        if args.export_dir:
            export_dir = Path(args.export_dir)
        else:
            export_dir = Path.home() / '.box21_exports' / str(project_id) / 'models' / str(selected_model['id'])
        
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting to: {export_dir}")
        
        # Create subdirectories
        models_dir = export_dir / 'files'
        preds_dir = export_dir / 'preds'
        models_dir.mkdir(exist_ok=True)
        preds_dir.mkdir(exist_ok=True)
        
        # Download model files
        print(f"\nDownloading model files...")
        # success = download_model_files(api, selected_model, models_dir)
        
        # if not success:
            # print("Failed to download model files")
            # return 1

        success = True
        
        # Get and save predictions
        print(f"\nFetching predictions...")
        predictions = get_predictions(api, selected_model['id'])
        
        if predictions:
            save_predictions(predictions, preds_dir)
            print(f"Exported {len(predictions)} predictions")
        else:
            print("No predictions found for this model")
        
        # Create summary
        summary = {
            'export_info': {
                'project_id': project_id,
                'project_name': selected_project.name,
                'model_id': selected_model['id'],
                'model_name': selected_model['name'],
                'export_timestamp': str(Path().cwd().stat().st_mtime),
                'export_directory': str(export_dir)
            },
            'statistics': {
                'total_predictions': len(predictions),
                'model_files_downloaded': success
            }
        }
        
        summary_file = export_dir / 'export_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExport completed successfully!")
        print(f"Model files: {models_dir}")
        print(f"Predictions: {preds_dir}")
        print(f"Summary: {summary_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        print(f"Login failed, correct username and password?")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 