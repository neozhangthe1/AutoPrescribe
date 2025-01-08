from astar_solver import generate_problems
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def analyze_solution_steps(problems):
    """Analyze solution steps for chain-of-thought and backtracking information"""
    results = {
        'total_problems': len(problems),
        'total_steps': 0,
        'backtrack_count': 0,
        'exploration_count': 0,
        'avg_steps_per_problem': 0.0,
        'max_steps': 0,
        'min_steps': float('inf'),
        'problems_with_backtracking': 0,
        'max_tokens_estimate': 0  # Rough estimate of tokens in longest solution
    }
    
    for problem in problems:
        steps = problem['steps']
        step_count = len(steps)
        results['total_steps'] += step_count
        results['max_steps'] = max(results['max_steps'], step_count)
        results['min_steps'] = min(results['min_steps'], step_count)
        
        # Count backtracking and exploration
        backtrack_steps = len([s for s in steps if 'Backtrack' in s])
        explore_steps = len([s for s in steps if 'Exploring' in s])
        results['backtrack_count'] += backtrack_steps
        results['exploration_count'] += explore_steps
        
        if backtrack_steps > 0:
            results['problems_with_backtracking'] += 1
        
        # Estimate tokens (rough approximation: 10 tokens per step on average)
        total_chars = sum(len(step) for step in steps)
        token_estimate = total_chars // 4  # Rough estimate: 4 chars per token
        results['max_tokens_estimate'] = max(
            results['max_tokens_estimate'], 
            token_estimate
        )
    
    # Calculate averages
    if len(problems) > 0:
        results['avg_steps_per_problem'] = results['total_steps'] / len(problems)
    
    return results

def plot_difficulty_distribution(difficulties, filename='difficulty_distribution.png'):
    """Plot and save difficulty distribution histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=20, edgecolor='black')
    plt.title('Distribution of Problem Difficulties')
    plt.xlabel('Difficulty')
    plt.ylabel('Number of Problems')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def main():
    # Generate full problem set with progress tracking
    print('Generating test problems...')
    problems = []
    target = 200  # Generate extra to ensure we get at least 100 after filtering
    
    print(f'Attempting to generate {target} problems...')
    for i in range(target):
        print(f'Generating problem {i+1}/{target}... ', end='')
        result = generate_problems(num_problems=1)
        if result:
            problems.extend(result)
            print(f'Success (difficulty: {result[0]["difficulty"]:.2f})')
        else:
            print('Failed')
            
    print(f'\nSuccessfully generated {len(problems)} problems')
    
    # Filter problems to ensure uniform difficulty distribution
    difficulties = [p['difficulty'] for p in problems]
    min_diff, max_diff = min(difficulties), max(difficulties)
    
    # Handle edge case where all difficulties are the same
    if max_diff - min_diff < 0.1:
        print("Warning: All problems have similar difficulty. Forcing distribution...")
        bucket_size = 2.0  # Force 5 buckets across 1-10 range
        min_diff = 1.0
        max_diff = 10.0
    else:
        bucket_size = (max_diff - min_diff) / 5
        
    buckets = {i: [] for i in range(5)}
    
    # Distribute problems into difficulty buckets
    for prob in problems:
        # Normalize difficulty to 0-1 range and multiply by number of buckets
        normalized = (prob['difficulty'] - min_diff) / (max_diff - min_diff)
        bucket = min(4, max(0, int(normalized * 5)))
        buckets[bucket].append(prob)
    
    # Select equal number of problems from each bucket
    balanced_problems = []
    target_per_bucket = 20  # Aim for 100 total problems
    for bucket in buckets.values():
        balanced_problems.extend(sorted(bucket, key=lambda x: len(x['steps']))[:target_per_bucket])
    
    problems = balanced_problems
    print(f'Selected {len(problems)} problems with balanced difficulty distribution')
    
    # Analyze results
    print('\nAnalyzing generated problems:')
    print(f'Total problems generated: {len(problems)}')
    
    difficulties = [p['difficulty'] for p in problems]
    print(f'Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}')
    print(f'Average difficulty: {sum(difficulties)/len(difficulties):.2f}')
    
    # Analyze solution steps
    analysis = analyze_solution_steps(problems)
    print('\nSolution analysis:')
    print(f'Total problems: {analysis["total_problems"]}')
    print(f'Average steps per solution: {analysis["avg_steps_per_problem"]:.2f}')
    print(f'Step range: {analysis["min_steps"]} - {analysis["max_steps"]} steps')
    print(f'Total backtrack operations: {analysis["backtrack_count"]}')
    print(f'Problems with backtracking: {analysis["problems_with_backtracking"]} ({analysis["problems_with_backtracking"]/len(problems)*100:.1f}%)')
    print(f'Total state explorations: {analysis["exploration_count"]}')
    print(f'\nContext length check:')
    print(f'Maximum tokens estimate: {analysis["max_tokens_estimate"]} tokens')
    print(f'Meets 4k+ requirement: {"Yes" if analysis["max_tokens_estimate"] >= 4000 else "No"}')
    
    # Plot difficulty distribution
    plot_difficulty_distribution(difficulties)
    print('\nDifficulty distribution plot saved as difficulty_distribution.png')
    
    # Print detailed example
    print('\nDetailed example of first problem:')
    example = problems[0]
    print(f'\nNumbers: {example["numbers"]}')
    print(f'Difficulty: {example["difficulty"]:.2f}')
    print('\nSolution steps:')
    for step in example['steps']:
        print(f'  {step}')
    
    # Save sample to file
    with open('sample_problems.json', 'w') as f:
        json.dump(problems, f, indent=2)
    print('\nSaved sample problems to sample_problems.json')

if __name__ == '__main__':
    main()
