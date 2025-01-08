"""
Test script for data generation functionality
"""
import random
from data_generator import (
    generate_dataset_with_distribution,
    save_dataset,
    load_dataset,
    calculate_difficulty
)
from search import solve_24_point

def test_single_problem():
    """Test solving a single 24-point problem with detailed steps"""
    numbers = [4, 8, 3, 6]
    success, steps = solve_24_point(numbers, max_iterations=2000, max_queue_size=5000)
    print(f"\nTesting with numbers: {numbers}")
    print(f"Solution found: {success}")
    if success:
        print("\nSolution steps:")
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")
        
        # Calculate and show difficulty
        difficulty = calculate_difficulty(numbers, steps)
        print(f"\nCalculated difficulty: {difficulty}")
        print("Analysis:")
        print(f"- Numbers: {numbers}")
        print(f"- Number of steps: {len(steps)}")
        print(f"- Operations used: {[step.split()[1] for step in steps]}")
        
    return success

def test_dataset_generation():
    """Test generating datasets with difficulty distribution"""
    import time
    start_time = time.time()
    print("\nGenerating test datasets...")
    
    # Test each difficulty level separately with more samples
    difficulty_levels = [
        ("Very Easy", 1.0, 25),
        ("Easy", 3.0, 25),
        ("Medium", 5.0, 25),
        ("Hard", 7.0, 25),
        ("Very Hard", 9.0, 25)
    ]
    
    all_problems = []
    success = True
    
    for level_name, target_diff, num_samples in difficulty_levels:
        print(f"\n=== Testing {level_name} Problem Generation ===")
        print(f"Attempting to generate {num_samples} problems (target difficulty: {target_diff})...")
        
        dataset = generate_dataset_with_distribution(
            num_samples=num_samples,
            timeout=60,
            batch_size=1,
            max_attempts=100,  # Increased max attempts
            target_difficulty=target_diff,
            require_solution=True
        )
        
        if len(dataset) < num_samples:
            print(f"❌ Failed to generate {level_name} problems")
            print(f"Generated {len(dataset)}/{num_samples} problems")
            if dataset:
                print("\nPartial results:")
                for i, prob in enumerate(dataset, 1):
                    print(f"\nProblem {i}:")
                    print(f"Numbers: {prob['numbers']}")
                    print(f"Steps: {prob['steps']}")
                    print(f"Difficulty: {prob['difficulty']:.2f}")
            success = False
            continue  # Continue testing other difficulties
        
        print(f"✓ Successfully generated {level_name} problems")
        print("\nGenerated problems:")
        for i, prob in enumerate(dataset, 1):
            print(f"\nProblem {i}:")
            print(f"Numbers: {prob['numbers']}")
            print(f"Steps: {prob['steps']}")
            print(f"Difficulty: {prob['difficulty']:.2f}")
        
        all_problems.extend(dataset)
    
    # Basic statistics on generated problems
    difficulties = [prob['difficulty'] for prob in all_problems]
    print("\nOverall Statistics:")
    print(f"Total problems generated: {len(all_problems)}")
    print(f"Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}")
    print(f"Average difficulty: {sum(difficulties)/len(difficulties):.2f}")
    
    # Test mixed dataset generation
    print("\nTesting mixed dataset generation (10 examples)...")
    mixed_dataset = generate_dataset_with_distribution(
        num_samples=10,
        timeout=60,
        batch_size=2,
        max_attempts=100
    )
    
    if len(mixed_dataset) < 10:
        print("❌ Failed to generate mixed dataset")
        return False
        
    print("✓ Successfully generated mixed dataset")
    print("\nSample problems from mixed dataset:")
    for i, example in enumerate(mixed_dataset[:3], 1):
        print(f"\nExample {i} (Difficulty: {example['difficulty']:.2f}):")
        print(f"Numbers: {example['numbers']}")
        print("Steps:")
        for j, step in enumerate(example['steps'], 1):
            print(f"  {j}. {step}")
    
    # Generate final datasets with progress reporting
    print("\nGenerating final datasets...")
    
    try:
        # Training set (150 examples for proper testing)
        print("\nGenerating training set (150 examples)...")
        train_dataset = generate_dataset_with_distribution(
            num_samples=150,
            timeout=180,
            batch_size=10,
            max_attempts=1000,
            allow_overflow=True,
            difficulty_tolerance=0.6  # Slightly increased tolerance
        )
        save_dataset(train_dataset, "train_dataset.json")
        print("✓ Training set generated and saved")
        
        # Test set (100 examples as required)
        print("\nGenerating test set (100 examples)...")
        test_dataset = generate_dataset_with_distribution(
            num_samples=100,
            timeout=120,
            batch_size=5,
            max_attempts=500
        )
        save_dataset(test_dataset, "test_dataset.json")
        print("✓ Test set generated and saved")
        
        # Analyze distributions
        train_difficulties = [ex['difficulty'] for ex in train_dataset]
        test_difficulties = [ex['difficulty'] for ex in test_dataset]
        
        print("\nDataset Statistics:")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        print(f"Training set difficulty range: {min(train_difficulties):.2f} - {max(train_difficulties):.2f}")
        print(f"Test set difficulty range: {min(test_difficulties):.2f} - {max(test_difficulties):.2f}")
        
        # Check distribution
        difficulty_ranges = [(0,2), (2,4), (4,6), (6,8), (8,10)]
        print("\nDifficulty Distribution:")
        
        def print_distribution(name, difficulties, total):
            print(f"\n{name}:")
            for low, high in difficulty_ranges:
                count = sum(1 for d in difficulties if low <= d < high)
                percentage = count/total*100 if total > 0 else 0
                print(f"Difficulty {low}-{high}: {count:3d} problems ({percentage:5.1f}%)")
                
        print_distribution("Training Set", train_difficulties, len(train_dataset))
        print_distribution("Test Set", test_difficulties, len(test_dataset))
        
        # Verify uniform distribution (chi-square test)
        def check_uniform_distribution(difficulties):
            counts = [sum(1 for d in difficulties if low <= d < high) 
                     for low, high in difficulty_ranges]
            expected = len(difficulties) / len(difficulty_ranges)
            chi_square = sum((obs - expected) ** 2 / expected for obs in counts)
            # Chi-square critical value for p=0.05, df=4 is 9.488
            return chi_square < 9.488
        
        train_uniform = check_uniform_distribution(train_difficulties)
        test_uniform = check_uniform_distribution(test_difficulties)
        
        print("\nUniformity Check:")
        print(f"Training set uniform: {'✓' if train_uniform else '✗'}")
        print(f"Test set uniform: {'✓' if test_uniform else '✗'}")
        
        # Basic validation passed if we got here
        return train_uniform and test_uniform and len(test_dataset) >= 100
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {str(e)}")
        return False

def test_solution_verification():
    """Test that solutions actually evaluate to 24"""
    def evaluate_expression(numbers, steps):
        """Evaluate if the solution steps actually result in 24"""
        # Create a copy of numbers to work with
        nums = numbers.copy()
        
        for step in steps:
            # Parse step like "Apply + to 9.0 and 13.0 to get 22.0"
            parts = step.split()
            op = parts[1]
            num1 = float(parts[3])
            num2 = float(parts[5])
            result = float(parts[8])
            
            # Verify numbers exist in our list
            if num1 not in nums or num2 not in nums:
                return False
                
            # Remove used numbers
            nums.remove(num1)
            nums.remove(num2)
            
            # Add result
            nums.append(result)
            
            # Verify the operation
            expected = None
            if op == '+':
                expected = num1 + num2
            elif op == '-':
                expected = num1 - num2
            elif op == '*':
                expected = num1 * num2
            elif op == '/' and abs(num2) > 1e-10:
                expected = num1 / num2
                
            if expected is None or abs(expected - result) > 1e-10:
                return False
        
        # Final verification
        return len(nums) == 1 and abs(nums[0] - 24) < 1e-10
    
    print("\nTesting solution verification...")
    test_problems = generate_dataset_with_distribution(
        num_samples=3,
        timeout=10,
        batch_size=1,
        max_attempts=10
    )
    all_valid = True
    
    for problem in test_problems:
        if not evaluate_expression(problem['numbers'], problem['steps']):
            print(f"Warning: Solution for {problem['numbers']} may not evaluate to 24")
            all_valid = False
    
    return all_valid

def generate_full_datasets():
    """Generate full training and test datasets using a two-phase approach:
    1. Generate a large pool of problems with lenient criteria
    2. Balance the pool to create uniform distributions
    """
    import os
    from generate_balanced_dataset import select_balanced_problems
    
    print("\n" + "="*60)
    print("Phase 1: Generating Problem Pool")
    print("="*60)
    
    # Generate a large pool of problems with very lenient settings
    problem_pool = generate_dataset_with_distribution(
        num_samples=500,  # Generate more problems than needed
        timeout=3600,  # 1 hour timeout
        batch_size=12,
        max_attempts=2000000,
        allow_overflow=True,
        difficulty_tolerance=8.0,  # Very lenient matching
        min_problems_per_bucket=5,  # Minimal requirements
        verbose=True,
        retry_backoff=2.0,
        max_retries=15,
        adaptive_tolerance=True,
        progressive_tightening=False,  # Don't tighten requirements
        save_intermediate=True,
        intermediate_file="intermediate_problems.json"
    )
    
    print(f"\nGenerated {len(problem_pool)} problems in pool")
    
    if len(problem_pool) < 250:  # Need at least this many for good distribution
        print("❌ Error: Not enough problems generated for balanced datasets")
        return False
        
    # Save the complete problem pool
    save_dataset(problem_pool, "problem_pool.json")
    print("\nProblem pool saved to problem_pool.json")
    
    print("\n" + "="*60)
    print("Phase 2: Creating Balanced Datasets")
    print("="*60)
    
    # Create training dataset (150 problems)
    print("\nCreating training dataset...")
    train_dataset = select_balanced_problems(problem_pool, 150)
    if len(train_dataset) < 150:
        print("❌ Error: Could not create balanced training dataset")
        return False
    
    # Remove training problems from pool
    train_signatures = {tuple(sorted(prob['numbers'])) for prob in train_dataset}
    remaining_pool = [p for p in problem_pool 
                     if tuple(sorted(p['numbers'])) not in train_signatures]
    
    # Create test dataset (100 problems)
    print("\nCreating test dataset...")
    test_dataset = select_balanced_problems(remaining_pool, 100)
    if len(test_dataset) < 100:
        print("❌ Error: Could not create balanced test dataset")
        return False
    
    # Save final datasets
    save_dataset(train_dataset, "train_dataset.json")
    save_dataset(test_dataset, "test_dataset.json")
    print("\nFinal datasets saved")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    def print_stats(name, dataset):
        difficulties = [prob['difficulty'] for prob in dataset]
        print(f"\n{name} Dataset:")
        print(f"Total problems: {len(dataset)}")
        print(f"Difficulty range: {min(difficulties):.1f} - {max(difficulties):.1f}")
        print(f"Average difficulty: {sum(difficulties)/len(difficulties):.1f}")
        print("\nDifficulty distribution:")
        for i in range(5):
            count = sum(1 for d in difficulties if i*2 <= d < (i+1)*2)
            print(f"  {i*2:2d}-{(i+1)*2:2d}: {count:3d} problems ({count/len(dataset)*100:5.1f}%)")
    
    print_stats("Problem Pool", problem_pool)
    print_stats("Training", train_dataset)
    print_stats("Test", test_dataset)
    
    # Verify uniformity using chi-square test
    def check_uniform_distribution(difficulties):
        ranges = [(0,2), (2,4), (4,6), (6,8), (8,10)]
        counts = [sum(1 for d in difficulties if low <= d < high) 
                 for low, high in ranges]
        expected = len(difficulties) / len(ranges)
        chi_square = sum((obs - expected) ** 2 / expected for obs in counts)
        return chi_square < 9.488  # Critical value for p=0.05, df=4
    
    train_difficulties = [prob['difficulty'] for prob in train_dataset]
    test_difficulties = [prob['difficulty'] for prob in test_dataset]
    
    train_uniform = check_uniform_distribution(train_difficulties)
    test_uniform = check_uniform_distribution(test_difficulties)
    
    print("\nUniformity Check:")
    print(f"Training set uniform: {'✓' if train_uniform else '✗'}")
    print(f"Test set uniform: {'✓' if test_uniform else '✗'}")
    
    return train_uniform and test_uniform and len(train_dataset) >= 150 and len(test_dataset) >= 100

def main():
    """Generate full datasets"""
    print("Starting full dataset generation...")
    success = generate_full_datasets()
    
    if success:
        print("\nDataset generation completed successfully!")
    else:
        print("\nDataset generation failed to meet size requirements.")

if __name__ == "__main__":
    main()
