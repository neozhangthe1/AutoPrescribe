# %% [markdown]
"""
# 24-Point Problem Solver - Dataset Generation

This notebook implements a dataset generator for 24-point arithmetic problems with difficulty balancing.

## Setup
First, we'll install required dependencies and set up our environment.
"""

# %% [code]
# Import required libraries
import random
import json
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# %% [markdown]
"""
## Import Dependencies
"""

# %% [code]
import random
import json
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from tqdm.notebook import tqdm

# %% [markdown]
"""
## Core Implementation
Here we define our core problem generation and difficulty calculation functions.
"""

# %% [code]
# Constants
OPERATIONS = ['+', '-', '*', '/']
MAX_ITERATIONS = 1000000
DIFFICULTY_RANGES = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

def calculate_difficulty(steps: List[str], numbers: List[int]) -> float:
    """Calculate problem difficulty based on operations and numbers used"""
    # Base difficulty from operations
    op_weights = {'+': 1.0, '-': 1.2, '*': 1.5, '/': 2.0}
    op_difficulty = sum(op_weights[step.split()[1]] for step in steps)
    
    # Number size complexity
    num_difficulty = sum(min(n, 13) / 2 for n in numbers) / len(numbers)
    
    # Solution length factor
    length_factor = len(steps) / 3.0
    
    # Operation diversity bonus
    unique_ops = len(set(step.split()[1] for step in steps))
    diversity_bonus = unique_ops * 0.5
    
    # Calculate final difficulty score
    difficulty = (op_difficulty * 0.4 + 
                 num_difficulty * 0.3 + 
                 length_factor * 0.2 + 
                 diversity_bonus * 0.1) * 2
    
    return min(10.0, max(1.0, difficulty))

def generate_problem(target_difficulty: Optional[float] = None,
                    difficulty_tolerance: float = 1.0) -> Dict:
    """Generate a single 24-point problem with optional target difficulty"""
    numbers = [random.randint(1, 13) for _ in range(4)]
    
    def evaluate(nums: List[float], ops: List[str]) -> Optional[float]:
        if len(nums) == 1:
            return float(nums[0])
        
        for i in range(len(nums) - 1):
            for op in ops:
                try:
                    num1, num2 = float(nums[i]), float(nums[i + 1])
                    if op == '+':
                        result = num1 + num2
                    elif op == '-':
                        result = num1 - num2
                    elif op == '*':
                        result = num1 * num2
                    elif op == '/' and num2 != 0:
                        result = num1 / num2
                    else:
                        continue
                        
                    new_nums = nums[:i] + [result] + nums[i + 2:]
                    sub_result = evaluate(new_nums, ops)
                    
                    if sub_result is not None and abs(sub_result - 24) < 1e-10:
                        return result
                except:
                    continue
        return None
    
    def solve() -> List[str]:
        steps = []
        current_nums = [float(n) for n in numbers]
        
        while len(current_nums) > 1:
            for i in range(len(current_nums) - 1):
                for op in OPERATIONS:
                    try:
                        num1, num2 = float(current_nums[i]), float(current_nums[i + 1])
                        if op == '+': result = num1 + num2
                        elif op == '-': result = num1 - num2
                        elif op == '*': result = num1 * num2
                        elif op == '/': result = num1 / num2
                        
                        new_nums = current_nums[:i] + [result] + current_nums[i + 2:]
                        if evaluate(new_nums, OPERATIONS) is not None:
                            steps.append(f"Apply {op} to {current_nums[i]} and {current_nums[i + 1]} to get {result}")
                            current_nums = new_nums
                            break
                    except:
                        continue
                if len(current_nums) < len(numbers):
                    break
            if len(steps) >= len(numbers) - 1:
                break
        return steps
    
    # Try to generate problem with target difficulty
    for _ in range(MAX_ITERATIONS):
        steps = solve()
        if steps:
            difficulty = calculate_difficulty(steps, numbers)
            if target_difficulty is None or abs(difficulty - target_difficulty) <= difficulty_tolerance:
                return {
                    'numbers': numbers,
                    'steps': steps,
                    'difficulty': difficulty,
                    'success': True
                }
        numbers = [random.randint(1, 13) for _ in range(4)]
    
    return {'success': False}

# %% [markdown]
"""
## Dataset Generation
Now we'll implement functions to generate balanced datasets of problems.
"""

# %% [code]
def generate_dataset(num_problems: int = 100, 
                    target_difficulty: Optional[float] = None,
                    difficulty_tolerance: float = 1.0,
                    max_duplicates: int = 3,
                    uniqueness_threshold: float = 0.7) -> List[Dict]:
    """Generate a dataset of 24-point problems"""
    problems = []
    unique_numbers = set()
    duplicates = 0
    
    with ThreadPoolExecutor() as executor:
        with tqdm(total=num_problems) as pbar:
            while len(problems) < num_problems and duplicates < max_duplicates:
                future = executor.submit(generate_problem, target_difficulty, difficulty_tolerance)
                result = future.result()
                
                if result['success']:
                    numbers = tuple(sorted(result['numbers']))
                    if numbers not in unique_numbers:
                        unique_numbers.add(numbers)
                        problems.append(result)
                        pbar.update(1)
                    else:
                        duplicates += 1
                        
                if duplicates >= max_duplicates and len(problems) / num_problems < uniqueness_threshold:
                    print(f"\nRestarting generation due to too many duplicates ({len(problems)}/{num_problems} problems generated)")
                    duplicates = 0
    
    return problems

# %% [markdown]
"""
## Generate Balanced Dataset
Let's generate a balanced dataset with problems across different difficulty ranges.
"""

# %% [code]
def generate_balanced_dataset(total_problems: int = 100) -> List[Dict]:
    """Generate a balanced dataset across difficulty ranges"""
    all_problems = []
    problems_per_range = total_problems // len(DIFFICULTY_RANGES)
    
    for diff_min, diff_max in DIFFICULTY_RANGES:
        print(f"\nGenerating problems for difficulty range {diff_min}-{diff_max}...")
        target_diff = (diff_min + diff_max) / 2
        tolerance = (diff_max - diff_min) / 2
        
        problems = generate_dataset(
            num_problems=problems_per_range,
            target_difficulty=target_diff,
            difficulty_tolerance=tolerance,
            max_duplicates=3,
            uniqueness_threshold=0.7
        )
        
        all_problems.extend(problems)
        
        # Print statistics for this range
        difficulties = [p['difficulty'] for p in problems]
        print(f"Generated {len(problems)} problems")
        print(f"Average difficulty: {np.mean(difficulties):.2f}")
        print(f"Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}")
    
    return all_problems

# %% [markdown]
"""
## Run Generation and Analysis
Generate the dataset and analyze its properties.
"""

# %% [code]
if __name__ == "__main__":
    # Generate dataset
    print("Generating balanced dataset...")
    all_problems = generate_balanced_dataset(100)
    
    # Save dataset
    with open('24_point_dataset.json', 'w') as f:
        json.dump(all_problems, f, indent=2)
    
    # Analyze results
    difficulties = [p['difficulty'] for p in all_problems]
    
    print("\nDataset Statistics:")
    print(f"Total problems: {len(all_problems)}")
    print(f"Average difficulty: {np.mean(difficulties):.2f}")
    print(f"Standard deviation: {np.std(difficulties):.2f}")
    print(f"Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}")
    
    print("\nDistribution across difficulty ranges:")
    for diff_min, diff_max in DIFFICULTY_RANGES:
        count = sum(1 for d in difficulties if diff_min <= d < diff_max)
        print(f"{diff_min}-{diff_max}: {count:3d} problems ({count/len(difficulties)*100:5.1f}%)")
