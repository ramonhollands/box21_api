# Box21 Export Tool

A command-line tool to export Box21 projects including assets, annotations, and labels to JSON format.

## Installation

### Option 1: Install as a command (recommended)

```bash
pip install -e .
```

This will install the `box21_export` command globally.

### Option 2: Run directly

```bash
python box21_export.py
```

## Usage

### Basic Usage

```bash
box21_export
```

This will start an interactive session where you'll be prompted for:
- Box21 project ID
- Box21 host (default: https://box21.ai)
- Username/email
- Password

You can also provide the project ID as an argument to skip the prompt:

```bash
box21_export PROJECT_ID
```

### Command Line Options

```bash
box21_export PROJECT_ID --host https://your-box21-instance.com --username your@email.com --batch-size 50 --export-dir /path/to/export
```

#### Options:
- `PROJECT_ID`: The Box21 project ID to export (required)
- `--host`: Box21 host URL (default: https://box21.ai)
- `--username`: Your Box21 username/email
- `--batch-size`: Number of assets to fetch per batch (default: 100)
- `--export-dir`: Custom export directory (default: ~/.box21_exports/project_id/)

## Export Structure

The tool exports data to the following structure:

```
~/.box21_exports/
└── {project_id}/
    ├── labels.json          # All project labels
    ├── assets.json          # All assets with annotations
    └── export_summary.json  # Export statistics and metadata
```

### Files Description

#### labels.json
Contains all labels in the project with their metadata and annotation statistics.

#### assets.json
Contains all assets with their metadata and associated annotations (bounding boxes and keypoints).

#### export_summary.json
Contains export statistics including:
- Project information
- Total counts (labels, assets, annotations)
- Validation statistics
- Export timestamp

## Features

- **Interactive Authentication**: Secure password input
- **Project Selection**: Choose from available projects
- **Batch Processing**: Efficient asset retrieval in configurable batches
- **Complete Data Export**: Labels, assets, and annotations
- **JSON Format**: Human-readable and machine-processable output
- **Progress Tracking**: Real-time progress updates during export

## Example Output

```
Box21 Export Tool
==================================================
Enter Box21 host (default: https://box21.ai): 
Enter username/email: user@example.com
Password: 

Connecting to https://box21.ai:443...

Exporting project: Project 123 (ID: 123)

Exporting to: /home/user/.box21_exports/123/

Exporting labels...
Exported 15 labels to /home/user/.box21_exports/123/labels.json

Exporting assets...
Fetching assets batch 1...
Exported 100 assets so far...
Fetching assets batch 2...
Exported 150 assets so far...
Exported 150 assets to /home/user/.box21_exports/123/assets.json

Export summary created: /home/user/.box21_exports/123/export_summary.json

Export completed successfully!
Total labels: 15
Total assets: 150
Export location: /home/user/.box21_exports/123/
```

## Requirements

- Python 3.7+
- requests
- PIL (Pillow)
- fastcore

## Troubleshooting

### Authentication Issues
- Verify your username/email and password
- Check if your Box21 instance requires different authentication

### Connection Issues
- Verify the host URL is correct
- Check network connectivity
- Ensure the Box21 instance is accessible

### Export Issues
- Check available disk space
- Verify write permissions to the export directory
- For large projects, consider using a smaller batch size 