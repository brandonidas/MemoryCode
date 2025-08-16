#!/usr/bin/env python3
"""
Extract Sub-5K Context Dialogues
Creates a subset containing only dialogues with ≤5000 characters
"""

import json
import sqlite3
import shutil
from pathlib import Path

def extract_tiny_subset(output_dir="dataset_tiny"):
    """Extract dialogues with ≤5000 characters to directory"""
    
    db_path = Path(__file__).parent / "subset_analysis.db"
    dataset_path = Path(__file__).parent.parent / "dataset"
    output_path = Path(__file__).parent.parent / output_dir
    
    if not db_path.exists():
        print("Error: subset_analysis.db not found. Run subset_analysis.py first.")
        return False
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get tiny dialogue IDs (≤5000 chars)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT dialogue_id FROM dialogue_profiles WHERE total_chars <= 5000 ORDER BY dialogue_id")
    dialogue_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Extracting {len(dialogue_ids)} tiny context dialogues (≤5K chars)")
    
    # Copy dialogue files
    copied_files = []
    missing_files = []
    
    for dialogue_id in dialogue_ids:
        source_file = dataset_path / f"{dialogue_id}.json"
        dest_file = output_path / f"{dialogue_id}.json"
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            copied_files.append(dialogue_id)
        else:
            missing_files.append(dialogue_id)
    
    # Create subset info file
    subset_info = {
        "subset_name": "tiny_context_sub5k",
        "creation_date": str(Path(__file__).parent.parent),
        "rationale": "All dialogues with ≤5000 characters for fast local model testing",
        "max_context_chars": 5000,
        "total_dialogues": len(dialogue_ids),
        "copied_dialogues": len(copied_files),
        "missing_dialogues": len(missing_files),
        "dialogue_ids": dialogue_ids,
        "source_dataset": str(dataset_path)
    }
    
    info_file = output_path / "subset_info.json"
    with open(info_file, 'w') as f:
        json.dump(subset_info, f, indent=2)
    
    conn.close()
    
    print(f"\n✅ Tiny subset extraction complete!")
    print(f"📁 Output directory: {output_path}")
    print(f"📄 Files copied: {len(copied_files)}")
    if missing_files:
        print(f"⚠️  Missing files: {len(missing_files)} - {missing_files}")
    print(f"ℹ️  Subset info saved to: {info_file}")
    
    return True

if __name__ == "__main__":
    extract_tiny_subset()
