#!/usr/bin/env python3

import subprocess
import sys
import os

def run_bert_training(source):
    """Run BERT training for a specific source"""
    print(f"\n{'='*50}")
    print(f"Training BERT for {source}")
    print(f"{'='*50}")
    
    cmd = [
        'python', 'scripts/finetune_bert_simple.py',
        '--source', source,
        '--epochs', '3',
        '--batch_size', '16',
        '--learning_rate', '2e-5'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training {source}: {e}")
        return False

def main():
    sources = ['reddit', 'x', 'news', 'meeting_minutes']
    
    print("Starting BERT training for all sources...")
    
    for source in sources:
        success = run_bert_training(source)
        if not success:
            print(f"Failed to train {source}, continuing with next source...")
    
    print("\nBERT training completed for all sources!")

if __name__ == "__main__":
    main() 