#!/usr/bin/env python3
"""Example: Collecting data with NWO policy for LeRobot training.

This shows how to use NWO for data collection, then export the dataset
for training with LeRobot's imitation learning.
"""

import os
import json
import requests

os.environ["NWO_API_KEY"] = "your_api_key_here"

from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig


def export_nwo_dataset(api_key, output_file="nwo_dataset.json"):
    """Export NWO API usage as training dataset.
    
    NWO logs all API calls with instructions, actions, and images.
    This exports them in LeRobot-compatible format.
    """
    endpoint = "https://nwo.capital/webapp/api-robotics.php"
    
    headers = {"X-API-Key": api_key}
    
    response = requests.get(
        f"{endpoint}?action=export_dataset",
        headers=headers,
        timeout=30,
    )
    
    if response.status_code == 200:
        dataset = response.json()
        
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Exported {dataset.get('total_records', 0)} records to {output_file}")
        return dataset
    else:
        print(f"✗ Export failed: {response.text}")
        return None


def convert_to_lerobot_format(nwo_dataset):
    """Convert NWO dataset to LeRobot dataset format.
    
    NWO format:
    {
        "version": "1.0",
        "data": [
            {
                "timestamp": "...",
                "instruction": "pick up the cube",
                "success": true,
                "image_url": "..."
            }
        ]
    }
    
    LeRobot format uses Parquet + video files.
    This is a simplified conversion showing the concept.
    """
    from pathlib import Path
    
    output_dir = Path("lerobot_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Create meta directory
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    # Write info.json (LeRobot metadata)
    info = {
        "codebase_version": "0.1.0",
        "robot_type": "nwo_compatible",
        "total_episodes": len(nwo_dataset.get("data", [])),
        "total_frames": len(nwo_dataset.get("data", [])),
        "total_tasks": len(set(d.get("instruction") for d in nwo_dataset.get("data", []))),
        "video": False,
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Write tasks.jsonl
    tasks = list(set(d.get("instruction") for d in nwo_dataset.get("data", [])))
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps({"task": task}) + "\n")
    
    print(f"✓ Converted dataset to LeRobot format in {output_dir}/")
    print(f"  - Episodes: {info['total_episodes']}")
    print(f"  - Unique tasks: {info['total_tasks']}")
    
    return output_dir


def main():
    print("=" * 60)
    print("NWO Dataset Export for LeRobot Training")
    print("=" * 60)
    
    api_key = os.getenv("NWO_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        print("\n⚠️  Please set NWO_API_KEY environment variable")
        return
    
    # Export dataset from NWO
    print("\n1. Exporting dataset from NWO API...")
    nwo_dataset = export_nwo_dataset(api_key)
    
    if nwo_dataset:
        # Show sample
        print("\n2. Sample records:")
        for record in nwo_dataset.get("data", [])[:3]:
            print(f"   - {record.get('instruction')[:50]}... (success: {record.get('success')})")
        
        # Convert to LeRobot format
        print("\n3. Converting to LeRobot format...")
        dataset_dir = convert_to_lerobot_format(nwo_dataset)
        
        print("\n" + "=" * 60)
        print("Next steps for training:")
        print("  1. Use LeRobot to load the dataset:")
        print(f"     dataset = LeRobotDataset('lerobot_dataset')")
        print("  2. Train a policy:")
        print("     python lerobot/scripts/train.py \\")
        print("       --dataset.dir=lerobot_dataset \\")
        print("       --policy.type=act \\")
        print("       --output_dir=./trained_policy")
        print("  3. Use trained policy for inference")
        print("=" * 60)


if __name__ == "__main__":
    main()
