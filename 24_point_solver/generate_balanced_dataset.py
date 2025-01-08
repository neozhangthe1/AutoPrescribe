"""
Utility module for creating balanced datasets from a pool of problems.
"""
import random
from typing import List, Dict, Tuple, Set
import json

def load_problems(filename: str) -> List[Dict]:
    """Load problems from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def select_balanced_problems(problems: List[Dict], target_size: int) -> List[Dict]:
    """
    Select problems to create a balanced dataset across difficulty levels.
    
    Args:
        problems: List of problem dictionaries with 'difficulty' key
        target_size: Desired size of the balanced dataset
        
    Returns:
        List of selected problems with approximately uniform difficulty distribution
    """
    # Group problems by difficulty bucket
    buckets = {i: [] for i in range(5)}  # 5 buckets for difficulties 0-2, 2-4, 4-6, 6-8, 8-10
    
    for problem in problems:
        difficulty = problem.get('difficulty', 0)
        bucket_idx = min(4, int(difficulty / 2))
        buckets[bucket_idx].append(problem)
    
    # Calculate target size per bucket
    target_per_bucket = max(target_size // 5, 1)
    
    # Print initial distribution
    print("\nInitial problem distribution:")
    for bucket_idx, bucket_problems in buckets.items():
        print(f"Difficulty {bucket_idx*2}-{(bucket_idx+1)*2}: {len(bucket_problems)} problems")
    
    # Select problems from each bucket
    selected_problems = []
    for bucket_idx, bucket_problems in buckets.items():
        # If we don't have enough problems in this bucket, take all of them
        num_to_select = min(target_per_bucket, len(bucket_problems))
        if num_to_select < target_per_bucket:
            print(f"\nWarning: Bucket {bucket_idx*2}-{(bucket_idx+1)*2} only has {len(bucket_problems)} problems")
            print(f"Will select all {num_to_select} problems from this bucket")
        
        # Prioritize problems with diverse operations
        bucket_problems.sort(
            key=lambda x: len(set(op for step in x['steps'] for op in step.split() if op in {'+', '-', '*', '/'})),
            reverse=True
        )
        
        selected = bucket_problems[:num_to_select]
        selected_problems.extend(selected)
        
        print(f"\nSelected {len(selected)} problems from bucket {bucket_idx*2}-{(bucket_idx+1)*2}")
        print(f"Average difficulty: {sum(p['difficulty'] for p in selected)/len(selected):.2f}")
    
    # Shuffle the final selection
    random.shuffle(selected_problems)
    
    print(f"\nTotal problems selected: {len(selected_problems)}")
    return selected_problems

def save_dataset(problems: List[Dict], filename: str):
    """Save the dataset to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(problems, f, indent=2)

def main():
    """Main function for command-line usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Create balanced datasets from a problem pool')
    parser.add_argument('input_file', help='Input JSON file containing the problem pool')
    parser.add_argument('output_file', help='Output JSON file for the balanced dataset')
    parser.add_argument('--size', type=int, default=100, help='Target size of the balanced dataset')
    args = parser.parse_args()
    
    problems = load_problems(args.input_file)
    print(f"Loaded {len(problems)} problems from {args.input_file}")
    
    balanced_problems = select_balanced_problems(problems, args.size)
    save_dataset(balanced_problems, args.output_file)
    print(f"Saved {len(balanced_problems)} problems to {args.output_file}")
    
    return True

if __name__ == "__main__":
    main()
