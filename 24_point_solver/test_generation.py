from data_generator import generate_dataset_with_distribution, save_dataset
import random

def test_dataset_generation():
    """Test generation of a balanced dataset with uniform difficulty distribution"""
    # Set random seed for reproducibility
    random.seed(42)
    
    print("\n" + "="*60)
    print("Testing Dataset Generation")
    print("="*60)
    
    # Generate full-size test dataset
    print('\nGenerating balanced dataset...')
    dataset = generate_dataset_with_distribution(
        num_samples=100,  # Target size for final dataset
        timeout=1800,     # 30 minutes timeout
        batch_size=12,    # Larger batch size for efficiency
        verbose=True,
        adaptive_tolerance=True,
        progressive_tightening=True,
        save_intermediate=True,
        allow_overflow=True,  # Allow more problems per bucket initially
        difficulty_tolerance=2.0,  # More lenient initial matching
        min_problems_per_bucket=15  # Ensure good distribution
    )

    # Save test dataset
    save_dataset(dataset, 'test_small.json')

    # Analyze dataset quality
    if not dataset:
        print("\nError: No dataset generated")
        return False
        
    difficulties = [p['difficulty'] for p in dataset]
    numbers_set = {tuple(sorted(p['numbers'])) for p in dataset}
    
    print('\nDataset Statistics:')
    print('-' * 40)
    print(f'Total problems: {len(dataset)}')
    print(f'Unique problems: {len(numbers_set)}')
    print(f'Difficulty range: {min(difficulties):.1f} - {max(difficulties):.1f}')
    
    # Analyze difficulty distribution
    print('\nDifficulty Distribution:')
    print('-' * 40)
    bucket_sizes = []
    target_per_bucket = len(dataset) // 5
    for i in range(5):
        count = sum(1 for d in difficulties if i*2 <= d < (i+1)*2)
        bucket_sizes.append(count)
        print(f'  {i*2:2d}-{(i+1)*2:2d}: {count:3d} problems ({count/len(dataset)*100:5.1f}%)')
    
    # Calculate distribution metrics
    mean_difficulty = sum(difficulties) / len(difficulties)
    variance = sum((d - mean_difficulty) ** 2 for d in difficulties) / len(difficulties)
    std_dev = variance ** 0.5
    
    print('\nDistribution Metrics:')
    print('-' * 40)
    print(f'Mean difficulty: {mean_difficulty:.2f}')
    print(f'Standard deviation: {std_dev:.2f}')
    
    # Verify requirements
    correct_size = len(dataset) >= 100
    all_unique = len(numbers_set) == len(dataset)
    min_bucket_size = min(bucket_sizes)
    max_bucket_size = max(bucket_sizes)
    uniform_distribution = (max_bucket_size - min_bucket_size) <= target_per_bucket * 0.4  # Allow 40% variation
    
    # Print validation results
    print('\nValidation Results:')
    print('-' * 40)
    print(f'✓ Size requirement (≥100): {"Pass" if correct_size else "Fail"}')
    print(f'✓ Uniqueness: {"Pass" if all_unique else "Fail"}')
    print(f'✓ Uniform distribution: {"Pass" if uniform_distribution else "Fail"}')
    
    if not uniform_distribution:
        print("\nDistribution Analysis:")
        print(f"Target problems per bucket: {target_per_bucket}")
        print(f"Current variation: {max_bucket_size - min_bucket_size}")
        print(f"Maximum allowed variation: {target_per_bucket * 0.4:.1f}")
    
    if not all_unique:
        print("\nDuplicate Analysis:")
        duplicates = {}
        for prob in dataset:
            nums = tuple(sorted(prob['numbers']))
            if nums not in duplicates:
                duplicates[nums] = []
            duplicates[nums].append(prob)
        
        print("\nDuplicate number combinations:")
        for nums, problems in duplicates.items():
            if len(problems) > 1:
                print(f"\nNumbers {list(nums)} appears {len(problems)} times:")
                for p in problems:
                    print(f"  Difficulty: {p['difficulty']:.2f}")
                    print(f"  Steps: {p['steps']}")
    
    # Save dataset if validation passes
    if correct_size and all_unique and uniform_distribution:
        print("\nSaving validated dataset...")
        save_dataset(dataset, 'validated_dataset.json')
        print("Dataset saved to validated_dataset.json")
    
    return correct_size and all_unique and uniform_distribution

if __name__ == '__main__':
    success = test_dataset_generation()
    print(f'\nTest {"passed" if success else "failed"}')
