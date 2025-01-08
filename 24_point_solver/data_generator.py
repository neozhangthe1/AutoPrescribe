"""
Generate training and test data for 24 point game with difficulty measurement
"""
import random
from typing import List, Tuple, Dict, Optional
import json
from search import solve_24_point
import math

def calculate_difficulty(numbers: List[int], steps: List[str]) -> float:
    """
    Calculate difficulty score based on:
    1. Operation complexity (primary factor)
    2. Number size (secondary factor)
    3. Solution length
    4. Pattern recognition (repeated numbers make it easier)
    
    Uses a more intuitive scale where:
    1.0-2.0: Very easy (single multiplication or simple chain)
    2.0-4.0: Easy (two operations, small numbers)
    4.0-6.0: Medium (three operations or larger numbers)
    6.0-8.0: Hard (complex operations, large numbers)
    8.0-10.0: Very hard (all operations, large numbers, complex steps)
    """
    # Start with zero base difficulty
    score = 0.0
    
    # Count operations
    op_counts = {'+': 0, '-': 0, '*': 0, '/': 0}
    for step in steps:
        op = step.split()[1]
        op_counts[op] += 1
    
    # Operation weights - multiplication and addition are easier
    total_ops = sum(op_counts.values())
    
    # Special case: Direct multiplication to 24
    if total_ops == 1 and op_counts['*'] == 1:
        # Check if it's a direct multiplication to 24
        nums = [float(x) for x in steps[0].split() if x.replace('.', '').isdigit()]
        if any(abs(a * b - 24.0) < 1e-6 for a in nums for b in nums if a != b):
            return 1.0  # Very easy case
    
    # Parse solution steps to understand the actual operations
    def parse_step(step):
        parts = step.split()
        op = parts[1]
        nums = [float(x) for x in parts if x.replace('.', '').isdigit()]
        return op, nums[:-1], nums[-1]  # op, operands, result
    
    parsed_steps = [parse_step(step) for step in steps]
    print(f"Parsed steps: {parsed_steps}")
    
    # Direct solutions (difficulty 1.0)
    if total_ops == 1:
        op, operands, result = parsed_steps[0]
        print(f"Checking direct solution - op: {op}, operands: {operands}, result: {result}")
        # Direct multiplication to 24
        if op == '*' and abs(result - 24.0) < 1e-6:
            print("Found direct multiplication to 24!")
            return 1.0
        # Direct addition to 24
        if op == '+' and abs(result - 24.0) < 1e-6:
            print("Found direct addition to 24!")
            return 1.0
    
    # Simple factor combinations with 1's
    if len(numbers) == 4 and numbers.count(1) >= 2:
        non_ones = [n for n in numbers if n != 1]
        print(f"Found numbers with 1's: non-ones = {non_ones}")
        if len(non_ones) == 2:
            print(f"Checking if {non_ones[0]} * {non_ones[1]} = 24")
            # Check if the solution only uses the non-ones to get 24
            used_nums = set()
            for _, operands, _ in parsed_steps:
                used_nums.update(operands)
            print(f"Used numbers in solution: {used_nums}")
            if all(n in used_nums for n in non_ones) and abs(non_ones[0] * non_ones[1] - 24.0) < 1e-6:
                print("Found simple factor combination!")
                return 1.0
    
    # Two-step solutions with small numbers
    if total_ops == 2 and max(numbers) <= 6:
        # All multiplication
        if all(op == '*' for op, _, _ in parsed_steps):
            return 1.2
        # Simple operations with small numbers
        return 1.5
    
    # Base difficulty calculation
    score = 0.0
    
    # Operation count contribution (minimal base scores)
    if total_ops == 1:
        score += 0.5
    elif total_ops == 2:
        score += 0.8
    elif total_ops == 3:
        score += 1.2
    else:  # 4 or more operations
        score += 1.8
    
    # Adjust score based on number sizes
    max_num = max(numbers)
    if max_num <= 6:
        score += 0.2
    elif max_num <= 9:
        score += 0.4
    else:
        score += 0.6
        
    # Special case for simple combinations
    if total_ops <= 2 and max(numbers) <= 6:
        return min(score + 0.3, 1.5)
    
    # Additional points for harder operations (reduced weights)
    score += op_counts['-'] * 0.4  # Subtraction is harder
    score += op_counts['/'] * 0.6  # Division is hardest
    score += op_counts['+'] * 0.2  # Addition is slightly harder than multiplication
    score += op_counts['*'] * 0.1  # Even multiplication adds some difficulty
    
    # Special case for simple combinations
    if total_ops <= 2 and max(numbers) <= 12 and all(n in {1,2,3,4,6,8,12} for n in numbers):
        score = min(score, 2.0)  # Cap score for simple factor combinations
    
    # Operation sequence complexity
    prev_op = None
    op_changes = 0
    for step in steps:
        op = step.split()[1]
        if prev_op and op != prev_op:
            op_changes += 1
        prev_op = op
    score += op_changes * 0.5  # Changing operations is harder
    
    # Number size and pattern factors
    numbers_set = set(numbers)
    max_num = max(numbers)
    min_num = min(numbers)
    
    # Size complexity
    if max_num > 10:
        score += 1.0
    elif max_num > 5:
        score += 0.5
    
    # Large number combinations are harder
    large_nums = sum(1 for n in numbers if n > 10)
    if large_nums > 1:
        score += large_nums * 0.5
    
    # Pattern recognition (repeated numbers make it easier)
    if len(numbers_set) == 4:  # All different
        score += 0.5
    elif len(numbers_set) == 1:  # All same
        score -= 0.5
    elif len(numbers_set) == 2 and numbers.count(list(numbers_set)[0]) == 2:  # Two pairs
        score -= 0.3
    
    # Special patterns that make it easier
    if 1 in numbers:
        score -= 0.2 * numbers.count(1)  # 1s make it easier
    if 2 in numbers:
        score -= 0.1 * numbers.count(2)  # 2s make it slightly easier
    
    # Factors of 24 make it easier
    factors = {1, 2, 3, 4, 6, 8, 12, 24}
    factor_count = sum(1 for n in numbers if n in factors)
    score -= 0.2 * factor_count
    
    # Operation variety increases difficulty
    unique_ops = sum(1 for count in op_counts.values() if count > 0)
    score += unique_ops * 0.5
    
    # Complex operation combinations
    if op_counts['/'] > 0 and op_counts['-'] > 0:
        score += 1.0  # Division and subtraction together is hard
    if op_counts['/'] > 1:
        score += 1.0  # Multiple divisions are very hard
    if op_counts['-'] > 1:
        score += 0.5  # Multiple subtractions are hard
        
    # Intermediate results far from 24 make it harder
    for step in steps:
        parts = step.split()
        result = float(parts[-1])
        if abs(result - 24) > 50:
            score += 0.5
    
    # Cap score range
    score = max(1.0, min(10.0, score))
    
    # Round to one decimal for cleaner difficulties
    return round(score, 1)

def generate_problem(numbers: Optional[List[int]] = None, target_difficulty: Optional[float] = None, difficulty_tolerance: float = 0.5, max_iterations: int = 1000, allow_overflow: bool = False, require_solution: bool = True) -> Dict:
    """
    Generate a 24 point problem with optional target difficulty
    Args:
        numbers: Optional list of 4 integers to use. If None, generates random numbers.
        target_difficulty: Optional target difficulty (0-10)
        difficulty_tolerance: How close to target difficulty we need to be
        max_iterations: Maximum number of attempts to generate a problem
        allow_overflow: Allow problems outside the target difficulty range
        require_solution: Only return problems that have a valid solution
    Returns:
        Dictionary with problem details
    """
    def generate_very_easy():
        """Generate problems with difficulty 0-2"""
        strategies = [
            # Direct multiplication to 24 with variety
            lambda: random.choice([
                [6, 4, 1, 1],  # 6 * 4 = 24
                [8, 3, 1, 1],  # 8 * 3 = 24
                [12, 2, 1, 1], # 12 * 2 = 24
                [4, 3, 2, 1],  # 4 * 3 * 2 = 24
                [6, 2, 2, 1],  # 6 * 2 * 2 = 24
                [3, 4, 2, 1],  # 3 * 4 * 2 = 24
                [2, 3, 4, 1],  # 2 * 3 * 4 = 24
                [24, 1, 1, 1]  # 24 * 1 = 24
            ]),
            # Simple addition patterns with variety
            lambda: random.choice([
                [6, 6, 6, 6],    # 6 + 6 + 6 + 6 = 24
                [8, 8, 8, 0],    # 8 + 8 + 8 + 0 = 24
                [12, 12, 0, 0],  # 12 + 12 + 0 + 0 = 24
                [7, 7, 5, 5],    # 7 + 7 + 5 + 5 = 24
                [10, 10, 2, 2],  # 10 + 10 + 2 + 2 = 24
                [15, 5, 2, 2]    # 15 + 5 + 2 + 2 = 24
            ]),
            # Mixed operations with small numbers
            lambda: random.choice([
                [4, 4, 4, 12],   # 4 + 4 + 4 + 12 = 24
                [3, 3, 3, 15],   # 3 + 3 + 3 + 15 = 24
                [5, 5, 5, 9],    # 5 + 5 + 5 + 9 = 24
                [2, 2, 10, 10],  # (2 + 2) * 10 = 40 - 16 = 24
                [6, 6, 6, 6]     # 6 * 6 - 6 - 6 = 24
            ]),
            # Generate random combinations of small numbers
            lambda: [random.randint(1, 6) for _ in range(4)]
        ]
        return random.choice(strategies)()

    def generate_easy():
        """Generate problems with difficulty 2-4"""
        strategies = [
            # Pairs that multiply close to 24 with variety
            lambda: list(random.choice([
                (4, 6),   # 24
                (3, 8),   # 24
                (2, 12),  # 24
                (4, 5),   # 20 (needs small adjustment)
                (3, 6),   # 18 (needs small adjustment)
                (5, 5),   # 25 (needs small adjustment)
                (7, 4),   # 28 (needs small adjustment)
                (9, 3)    # 27 (needs small adjustment)
            ])) + [random.randint(1, 4) for _ in range(2)],
            # Numbers that add to 24 with adjustments
            lambda: sorted([
                random.randint(5, 9),
                random.randint(4, 8),
                random.randint(3, 7),
                random.randint(2, 6)
            ]),
            # Mixed operations with small-medium numbers
            lambda: [
                random.randint(3, 8),
                random.randint(2, 7),
                random.randint(2, 6),
                random.randint(1, 5)
            ],
            # Numbers requiring subtraction
            lambda: [
                random.randint(8, 12),
                random.randint(4, 8),
                random.randint(2, 6),
                random.randint(1, 4)
            ],
            # Numbers requiring division
            lambda: sorted([
                random.randint(6, 12),
                random.randint(4, 8),
                random.randint(2, 6),
                random.randint(1, 4)
            ], reverse=True)
        ]
        return random.choice(strategies)()

    def generate_medium():
        """Generate problems with difficulty 4-6"""
        strategies = [
            # One large number with varied smaller numbers
            lambda: [
                random.randint(8, 12),  # Large number
                random.randint(4, 7),   # Medium number
                random.randint(3, 6),   # Small-medium number
                random.randint(2, 5)    # Small number
            ],
            # Two medium pairs with variety
            lambda: [
                random.randint(6, 9),   # First medium-large
                random.randint(5, 8),   # Second medium-large
                random.randint(3, 6),   # First medium-small
                random.randint(2, 5)    # Second medium-small
            ],
            # Numbers requiring division with variety
            lambda: sorted([
                random.randint(8, 13),  # Large dividend
                random.randint(4, 8),   # Medium divisor
                random.randint(3, 6),   # Additional number
                random.randint(2, 5)    # Small number
            ], reverse=True),
            # Mixed operations requiring planning
            lambda: [
                random.randint(7, 11),  # Larger number
                random.randint(5, 9),   # Medium-large number
                random.randint(3, 7),   # Medium-small number
                random.randint(2, 5)    # Small number
            ],
            # Numbers requiring multiple operations
            lambda: sorted([
                random.randint(6, 10),
                random.randint(5, 9),
                random.randint(4, 8),
                random.randint(3, 7)
            ], reverse=True)
        ]
        return random.choice(strategies)()

    def generate_hard():
        """Generate problems with difficulty 6-8"""
        strategies = [
            # Two large numbers with medium numbers
            lambda: [
                random.randint(10, 13),  # First large number
                random.randint(8, 12),   # Second large number
                random.randint(4, 7),    # Medium number
                random.randint(2, 5)     # Small number
            ],
            # Numbers requiring multiple divisions
            lambda: sorted([
                random.randint(10, 15),  # Large dividend
                random.randint(6, 10),   # Medium-large divisor
                random.randint(4, 8),    # Medium divisor
                random.randint(2, 6)     # Small number
            ], reverse=True),
            # Mixed large and small requiring complex operations
            lambda: [
                random.randint(9, 13),   # Large number
                random.randint(6, 10),   # Medium-large number
                random.randint(3, 7),    # Medium number
                random.randint(2, 5)     # Small number
            ],
            # Numbers requiring multiple subtractions
            lambda: sorted([
                random.randint(11, 15),  # Very large number
                random.randint(7, 11),   # Large number
                random.randint(4, 8),    # Medium number
                random.randint(2, 6)     # Small number
            ], reverse=True),
            # Numbers requiring mixed operations
            lambda: [
                random.randint(8, 12),   # Large number
                random.randint(7, 11),   # Another large number
                random.randint(5, 9),    # Medium number
                random.randint(3, 7)     # Small-medium number
            ]
        ]
        return random.choice(strategies)()

    def generate_very_hard():
        """Generate problems with difficulty 8-10"""
        strategies = [
            # Three large numbers with one medium
            lambda: [
                random.randint(11, 15),  # Very large number
                random.randint(10, 14),  # Large number
                random.randint(9, 13),   # Another large number
                random.randint(5, 9)     # Medium number
            ],
            # All large numbers with variety
            lambda: sorted([
                random.randint(12, 15),  # Largest number
                random.randint(10, 13),  # Second largest
                random.randint(8, 11),   # Third largest
                random.randint(6, 9)     # Fourth largest
            ], reverse=True),
            # Complex division patterns
            lambda: sorted([
                random.randint(13, 17),  # Very large dividend
                random.randint(9, 13),   # Large divisor
                random.randint(7, 11),   # Medium-large number
                random.randint(5, 9)     # Medium number
            ], reverse=True),
            # Mixed operations with large numbers
            lambda: [
                random.randint(10, 15),  # Large number
                random.randint(9, 14),   # Another large number
                random.randint(8, 13),   # Third large number
                random.randint(7, 12)    # Fourth large number
            ],
            # Numbers requiring multiple complex operations
            lambda: sorted([
                random.randint(11, 16),  # Very large number
                random.randint(9, 14),   # Large number
                random.randint(7, 12),   # Medium-large number
                random.randint(5, 10)    # Medium number
            ], reverse=True)
        ]
        return random.choice(strategies)()

    # Track recently used combinations to prevent repetition
    if not hasattr(generate_problem, 'recent_combinations'):
        generate_problem.recent_combinations = set()
        
    def is_similar_combination(numbers):
        """Check if numbers are too similar to recent combinations"""
        sorted_nums = tuple(sorted(numbers))
        
        # Check if exact combination was recently used
        if sorted_nums in generate_problem.recent_combinations:
            return True
            
        # Check for patterns like all same numbers
        if len(set(numbers)) == 1:
            return True
            
        # Check for patterns with too many zeros or ones
        if numbers.count(0) > 1 or numbers.count(1) > 2:
            return True
            
        # Check for patterns with too many similar numbers
        counts = {}
        for n in numbers:
            counts[n] = counts.get(n, 0) + 1
            if counts[n] > 2:  # No more than 2 of any number
                return True
                
        return False
        
    def add_to_recent(numbers):
        """Add numbers to recent combinations"""
        sorted_nums = tuple(sorted(numbers))
        generate_problem.recent_combinations.add(sorted_nums)
        if len(generate_problem.recent_combinations) > 100:  # Keep last 100
            generate_problem.recent_combinations.pop()
            
    # Use provided numbers or generate new ones
    if numbers is not None:
        # Verify the numbers are valid
        if len(numbers) != 4 or not all(isinstance(n, int) and n > 0 for n in numbers):
            return {'success': False, 'error': 'Invalid numbers provided'}
        
        # Try to solve with these numbers
        success, steps = solve_24_point(numbers)
        if success or not require_solution:
            difficulty = calculate_difficulty(numbers, steps)
            # Check if difficulty matches target if specified
            if target_difficulty is not None and not allow_overflow:
                if abs(difficulty - target_difficulty) > difficulty_tolerance:
                    return {'success': False, 'error': 'Difficulty outside target range'}
            return {
                'numbers': numbers,
                'steps': steps,
                'difficulty': difficulty,
                'success': success
            }
        return {'success': False, 'error': 'No solution found'}
    
    # Generate new numbers with pattern prevention
    max_attempts = max_iterations
    for attempt in range(max_attempts):
        # Select generation strategy based on target difficulty
        if target_difficulty is not None:
            if target_difficulty < 2:
                candidate_numbers = generate_very_easy()
            elif target_difficulty < 4:
                candidate_numbers = generate_easy()
            elif target_difficulty < 6:
                candidate_numbers = generate_medium()
            elif target_difficulty < 8:
                candidate_numbers = generate_hard()
            else:
                candidate_numbers = generate_very_hard()
        else:
            # Random difficulty with weighted distribution
            weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Favor medium difficulty
            strategy = random.choices(
                [generate_very_easy, generate_easy, generate_medium, generate_hard, generate_very_hard],
                weights=weights
            )[0]
            candidate_numbers = strategy()
            
        if not is_similar_combination(candidate_numbers):
            random.shuffle(candidate_numbers)
            success, steps = solve_24_point(candidate_numbers)
            if success:
                difficulty = calculate_difficulty(candidate_numbers, steps)
                # Check if difficulty matches target if specified
                if target_difficulty is not None and not allow_overflow:
                    if abs(difficulty - target_difficulty) > difficulty_tolerance:
                        continue
                add_to_recent(candidate_numbers)
                return {
                    'numbers': candidate_numbers,
                    'steps': steps,
                    'difficulty': difficulty,
                    'success': True
                }
            
    # If all attempts failed, return failure
    return {'success': False, 'error': 'Could not generate valid problem'}

def generate_dataset_with_distribution(
    num_samples: int,
    require_solution: bool = True,
    timeout: int = 1800,  # 30 minutes timeout
    batch_size: int = 12,  # Much larger batch size
    max_attempts: int = 1000000,  # Very high max attempts
    target_difficulty: Optional[float] = None,  # Optional target difficulty
    allow_overflow: bool = True,  # Allow more problems per bucket
    difficulty_tolerance: float = 6.0,  # Extremely lenient initial matching
    min_problems_per_bucket: int = 10,  # Minimal bucket requirement
    verbose: bool = False,  # Control debug output
    retry_backoff: float = 2.5,  # More aggressive backoff
    max_retries: int = 12,  # More retries allowed
    adaptive_tolerance: bool = True,  # Enable adaptive tolerance
    progressive_tightening: bool = True,  # Enable progressive tightening
    save_intermediate: bool = False,  # Save intermediate results
    intermediate_file: str = "intermediate_problems.json"  # File to save intermediate results
) -> List[Dict]:
    """
    Generate dataset with uniform difficulty distribution
    Args:
        num_samples: Total number of samples to generate
        require_solution: Only include problems with solutions
        timeout: Maximum time in seconds to spend generating each batch
        batch_size: Number of problems to generate in parallel
        max_attempts: Maximum number of attempts per difficulty bucket
    Returns:
        List of {numbers, steps, difficulty} dictionaries
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait
    import time
    from tqdm import tqdm
    import random  # Ensure random is imported
    
    def generate_single_problem(target_diff: Optional[float] = None):
        try:
            # Generate problem with target difficulty if specified
            result = generate_problem(
                target_difficulty=target_diff,
                difficulty_tolerance=difficulty_tolerance,
                max_iterations=1000,
                allow_overflow=True
            )
            if result and result['success']:
                if verbose:
                    print(f"Generated valid problem: {result}")
                return result
            return None
            
            print("No solution found")  # Debug output
            return None
            
        except Exception as e:
            print(f"Error in generate_single_problem: {str(e)}")  # Debug output
            return None
    
    # Define difficulty ranges with overlapping boundaries for smoother distribution
    bucket_difficulties = {
        0: (0.0, 2.5),   # Very easy (expanded range)
        1: (2.0, 4.5),   # Easy (expanded range)
        2: (4.0, 6.5),   # Medium (expanded range)
        3: (5.5, 8.5),   # Hard (expanded range)
        4: (7.5, 10.0)   # Very hard (expanded range)
    }
    
    def generate_batch(target_diff: Optional[float] = None):
        if verbose:
            print(f"\nStarting batch generation with target difficulty: {target_diff}")
        batch_results = []
        local_seen = set()  # Track signatures within this batch
        current_tolerance = difficulty_tolerance
        retry_count = 0
        total_problems = len(dataset)
        batch_start_time = time.time()
        
        # Update bucket priorities based on current distribution
        for i in range(5):
            if bucket_counts[i] < target_per_bucket:
                # Higher priority for empty buckets
                bucket_priorities[i] = 2.0 + (target_per_bucket - bucket_counts[i]) / target_per_bucket
            else:
                # Lower priority for filled buckets
                bucket_priorities[i] = max(0.5, 1.0 - (bucket_counts[i] - target_per_bucket) / target_per_bucket)
        
        # Generate problems in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            while (len(batch_results) < batch_size and 
                   len(local_seen) < batch_size * 12 and 
                   time.time() - batch_start_time < timeout / 10):  # Timeout for batch
                
                # Adjust tolerance if adaptive mode is enabled
                if adaptive_tolerance and retry_count > 0:
                    current_tolerance = min(8.0, difficulty_tolerance * (1 + 0.5 * retry_count))
                
                # Submit a wave of jobs
                wave_futures = []
                wave_size = min(batch_size * 3, batch_size * 8 - len(local_seen))
                
                for _ in range(wave_size):
                    # Get dynamic target difficulty if none provided
                    if target_diff is None:
                        # Use bucket priorities to guide difficulty targeting
                        empty_buckets = [i for i, count in bucket_counts.items() if count < target_per_bucket]
                        if empty_buckets and random.random() < 0.8:  # 80% chance to target empty buckets
                            weights = [bucket_priorities[i] for i in empty_buckets]
                            bucket = random.choices(empty_buckets, weights=weights)[0]
                            min_diff, max_diff = bucket_difficulties[bucket]
                            adjusted_diff = random.uniform(min_diff, max_diff)
                        else:
                            adjusted_diff = get_target_difficulty()
                    else:
                        # Add increasing randomness based on retry count
                        base_variance = min(1.0, current_tolerance / 2)
                        variance_scale = 1.0 + (retry_count * 0.2)  # Increase variance with retries
                        adjusted_diff = target_diff + random.uniform(-base_variance * variance_scale, base_variance * variance_scale)
                    
                    wave_futures.append(executor.submit(generate_single_problem, adjusted_diff))
                
                # Process results as they complete with timeout
                try:
                    done_futures, _ = wait(wave_futures, timeout=timeout/20)  # Timeout for wave
                    for future in done_futures:
                        try:
                            result = future.result(timeout=1)  # Timeout for individual result
                            if result is not None:
                                difficulty = result['difficulty']
                                numbers_tuple = tuple(sorted(result['numbers']))
                                
                                # Skip if we've seen this problem before
                                if numbers_tuple in seen_signatures or numbers_tuple in local_seen:
                                    if verbose:
                                        print("Skipping duplicate problem")
                                    continue
                                
                                # Check if difficulty is within tolerance or matches any bucket needs
                                valid_difficulty = False
                                if target_diff is None:
                                    # Check if this difficulty helps fill any needed bucket
                                    for bucket, (min_diff, max_diff) in bucket_difficulties.items():
                                        if min_diff <= difficulty <= max_diff and bucket_counts[bucket] < max_per_bucket:
                                            valid_difficulty = True
                                            # Update bucket priority
                                            bucket_priorities[bucket] *= 0.9  # Reduce priority as we fill the bucket
                                            break
                                else:
                                    valid_difficulty = abs(difficulty - target_diff) <= current_tolerance
                                
                                if valid_difficulty:
                                    batch_results.append(result)
                                    local_seen.add(numbers_tuple)
                                    if len(batch_results) >= batch_size:
                                        break
                                elif verbose:
                                    print(f"Difficulty {difficulty:.1f} outside target range")
                        except TimeoutError:
                            if verbose:
                                print("Problem generation timed out")
                            continue
                        except Exception as e:
                            if verbose:
                                print(f"Error processing result: {str(e)}")
                            continue
                except Exception as e:
                    if verbose:
                        print(f"Error in wave processing: {str(e)}")
                
                retry_count += 1
                if retry_count >= max_retries:
                    break
                
                # Print progress every 30 seconds
                if verbose and time.time() - batch_start_time > 30:
                    print(f"\nBatch Progress:")
                    print(f"Generated {len(batch_results)}/{batch_size} problems")
                    print(f"Current bucket counts:", {i: bucket_counts[i] for i in range(5)})
                    print(f"Current bucket priorities:", {i: f"{p:.2f}" for i, p in bucket_priorities.items()})
        
        if verbose:
            print(f"Batch generation complete. Generated {len(batch_results)} valid problems")
            print(f"Time taken: {time.time() - batch_start_time:.1f}s")
        
        return batch_results
    
    # Initialize dataset and tracking variables
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time  # Add time import for progress tracking
    import os
    import json
    
    dataset = []  # Initialize empty dataset
    target_per_bucket = max(num_samples // 5, min_problems_per_bucket)
    max_per_bucket = int(target_per_bucket * 2.5) if allow_overflow else target_per_bucket  # Allow more overflow
    buckets = {i: [] for i in range(5)}  # 0-2, 2-4, 4-6, 6-8, 8-10
    bucket_counts = {i: 0 for i in range(5)}  # Track counts separately
    seen_signatures = set()  # Track unique problem signatures
    min_problems_needed = num_samples  # Minimum total problems needed
    attempts = 0
    
    # Load intermediate results if they exist
    if save_intermediate and os.path.exists(intermediate_file):
        try:
            with open(intermediate_file, 'r') as f:
                intermediate_data = json.load(f)
                dataset.extend(intermediate_data)
                print(f"Loaded {len(intermediate_data)} problems from intermediate file")
                # Update tracking sets and buckets
                for problem in intermediate_data:
                    numbers_tuple = tuple(sorted(problem['numbers']))
                    seen_signatures.add(numbers_tuple)
                    if 'difficulty' in problem:
                        bucket_idx = min(4, int(problem['difficulty'] / 2))
                        buckets[bucket_idx].append(problem)
                        bucket_counts[bucket_idx] += 1
        except Exception as e:
            print(f"Error loading intermediate results: {e}")
    # Initialize core variables
    target_per_bucket = max(num_samples // 5, min_problems_per_bucket)
    max_per_bucket = int(target_per_bucket * 2.5) if allow_overflow else target_per_bucket
    buckets = {i: [] for i in range(5)}  # 0-2, 2-4, 4-6, 6-8, 8-10
    bucket_counts = {i: 0 for i in range(5)}
    seen_signatures = set()
    min_problems_needed = num_samples
    attempts = 0
    
    # Initialize timing variables
    start_time = time.time()
    last_progress_time = start_time
    
    # Initialize bucket priorities - higher priority for empty buckets
    bucket_priorities = {i: 2.0 for i in range(5)}
    
    # Dynamic difficulty targeting with overlapping ranges for smoother distribution
    bucket_difficulties = {
        0: (0.0, 2.5),   # Very easy (expanded range)
        1: (2.0, 4.5),   # Easy (expanded range)
        2: (4.0, 6.5),   # Medium (expanded range)
        3: (5.5, 8.5),   # Hard (expanded range)
        4: (7.5, 10.0)   # Very hard (expanded range)
    }
    
    def get_target_difficulty():
        """Get target difficulty based on current distribution with fallback"""
        # Find buckets that need more problems
        needed_buckets = [
            i for i, count in bucket_counts.items()
            if count < target_per_bucket
        ]
        if not needed_buckets:
            return None
            
        # First try: Target specific bucket
        if random.random() < 0.7:  # 70% chance to target specific bucket
            # Prioritize empty buckets
            empty_buckets = [i for i in needed_buckets if bucket_counts[i] == 0]
            if empty_buckets:
                bucket = random.choice(empty_buckets)
            else:
                # Weight buckets by how many more problems they need
                weights = [
                    1.0 - (bucket_counts[i] / target_per_bucket)
                    for i in needed_buckets
                ]
                bucket = random.choices(needed_buckets, weights=weights)[0]
                
            # Return random difficulty within bucket range
            min_diff, max_diff = bucket_difficulties[bucket]
            return random.uniform(min_diff, max_diff)
        
        # Fallback: Generate any valid difficulty
        else:  # 30% chance to be more flexible
            return random.uniform(0.5, 9.5)  # Full range
    
    print("\n" + "="*60)
    print(f"Generating {num_samples} problems with uniform difficulty distribution")
    print(f"Max attempts: {max_attempts}, Timeout: {timeout}s, Batch size: {batch_size}")
    print(f"Target difficulty: {target_difficulty if target_difficulty is not None else 'auto'}")
    print("="*60 + "\n")
    
    total_generated = 0
    difficulty_tolerance = 1.0  # Allow problems within ±1.0 of target difficulty
    
    while total_generated < num_samples and attempts < max_attempts:
        attempts += 1
        print(f"\nAttempt {attempts}/{max_attempts}")
        
        # Generate problems in parallel
        if target_difficulty is not None:
            print(f"Targeting specific difficulty: {target_difficulty}")
            batch_results = generate_batch(target_difficulty)
            # Filter results by difficulty tolerance
            batch_results = [
                result for result in batch_results 
                if abs(result['difficulty'] - target_difficulty) <= difficulty_tolerance
            ]
        else:
            # Target multiple buckets that need more problems
            needed_buckets = sorted(
                [(i, c) for i, c in bucket_counts.items() if c < max_per_bucket],
                key=lambda x: x[1]
            )[:batch_size]
            
            if not needed_buckets:
                print("All buckets are full enough")
                break
                
            print(f"Targeting buckets: {needed_buckets}")
            target_diffs = [i * 2 + 1 for i, _ in needed_buckets]  # Middle of each bucket range
            
            # Generate problems for each target difficulty
            all_results = []
            for target_diff in target_diffs:
                results = generate_batch(target_diff)
                all_results.extend(results)
            
            # Process all generated problems
            batch_results = []
            for result in all_results:
                if result:
                    difficulty = result['difficulty']
                    bucket_idx = min(4, int(difficulty / 2))
                    
                    # Check if problem signature is unique
                    signature = tuple(sorted(result['numbers']))
                    if signature in seen_signatures:
                        print(f"Duplicate problem signature: {signature}")
                        continue
                        
                    # Try to fit in appropriate bucket
                    if bucket_counts[bucket_idx] < max_per_bucket:
                        batch_results.append(result)
                        seen_signatures.add(signature)
                    else:
                        print(f"Bucket {bucket_idx} is full")
        
        print(f"Got {len(batch_results)} results")
        
        for result in batch_results:
            if result is None or (not result['success'] and require_solution):
                print("Skipping unsuccessful result")
                continue
                
            difficulty = result['difficulty']
            bucket_idx = min(4, int(difficulty / 2))
            print(f"Result difficulty: {difficulty}, assigning to bucket {bucket_idx}")
            
            if bucket_counts[bucket_idx] < target_per_bucket:
                buckets[bucket_idx].append(result)
                bucket_counts[bucket_idx] += 1
                total_generated += 1
                print("\n✓ Successfully added problem:")
                print(f"  Numbers: {result['numbers']}")
                print(f"  Steps: {result['steps']}")
                print(f"  Difficulty: {result['difficulty']:.2f}")
                print(f"  Bucket: {bucket_idx} ({bucket_idx*2}-{(bucket_idx+1)*2})")
                print(f"  Progress: {total_generated}/{num_samples}")
                
                # Print progress with timing
                current_time = time.time()
                elapsed = current_time - start_time
                rate = total_generated / elapsed if elapsed > 0 else 0
                eta = (num_samples - total_generated) / rate if rate > 0 else float('inf')
                
                if current_time - last_progress_time >= 30:  # Update every 30 seconds
                    print(f"\nProgress Update (after {elapsed:.1f}s):")
                    print(f"Generation rate: {rate:.2f} problems/second")
                    print(f"Estimated time remaining: {eta:.1f}s")
                    last_progress_time = current_time
                    
                    # Save intermediate results every 30 seconds if enabled
                    if save_intermediate:
                        try:
                            with open(intermediate_file, 'w') as f:
                                json.dump(dataset, f, indent=2)
                            print(f"Saved {len(dataset)} problems to {intermediate_file}")
                        except Exception as e:
                            print(f"Error saving intermediate results: {e}")
                
                # Print current distribution
                print("\nCurrent distribution:")
                for i in range(5):
                    count = bucket_counts[i]
                    target = target_per_bucket
                    print(f"  Difficulty {i*2:2d}-{(i+1)*2:2d}: {count:3d}/{target:3d} problems ({count/target*100:5.1f}%)")
    
    if attempts >= max_attempts:
        print("\n" + "!"*60)
        print("WARNING: Reached maximum attempts, dataset may not have perfect distribution")
        print("!"*60)
    
    print("\n" + "="*60)
    print("Final Dataset Statistics")
    print("="*60)
    print(f"Total problems generated: {total_generated}/{num_samples}")
    print(f"Total attempts: {attempts}")
    print("\nDifficulty distribution:")
    for i, bucket in buckets.items():
        print(f"  Difficulty {i*2:2d}-{(i+1)*2:2d}: {len(bucket):3d} problems")
    
    # Balance and combine buckets
    dataset = []
    target_per_bucket = num_samples // 5  # We want exactly equal buckets
    
    print("\nBalancing buckets to achieve uniform distribution...")
    print(f"Target problems per bucket: {target_per_bucket}")
    
    # Score problems by operation diversity and solution length
    def score_problem(problem):
        # Count unique operations
        ops = set(op for step in problem['steps'] for op in step.split() if op in {'+', '-', '*', '/'})
        op_diversity = len(ops)
        # Prefer shorter solutions (normalized to 0-1 range)
        solution_length = len(problem['steps'])
        length_score = 1.0 / solution_length
        # Combine scores (weight diversity more heavily)
        return op_diversity * 0.7 + length_score * 0.3
    
    # First, pad all buckets to target size
    for difficulty in range(5):
        bucket = buckets[difficulty]
        if not bucket:  # If bucket is empty, try to fill it with problems from adjacent buckets
            print(f"\nWarning: Bucket {difficulty*2}-{(difficulty+1)*2} is empty")
            # Try to get problems from adjacent buckets
            adjacent_problems = []
            if difficulty > 0:
                adjacent_problems.extend(buckets[difficulty - 1])
            if difficulty < 4:
                adjacent_problems.extend(buckets[difficulty + 1])
            if adjacent_problems:
                # Sort by how close they are to this bucket's target difficulty
                target_diff = difficulty * 2 + 1  # Middle of bucket
                adjacent_problems.sort(key=lambda p: abs(p['difficulty'] - target_diff))
                bucket.extend(adjacent_problems[:target_per_bucket])
        
        if len(bucket) < target_per_bucket:
            print(f"\nWarning: Bucket {difficulty*2}-{(difficulty+1)*2} only has {len(bucket)} problems")
            print("Duplicating best problems to reach target size")
            # Sort problems by score before duplicating
            if bucket:
                bucket.sort(key=score_problem, reverse=True)
                while len(bucket) < target_per_bucket:
                    # Create more varied problems when padding
                    base_problem = random.choice(bucket[:max(1, len(bucket)//2)])  # Prefer better problems
                    
                    # Create a new problem with similar structure but different numbers
                    new_numbers = list(base_problem['numbers'])
                    for i in range(len(new_numbers)):
                        # Adjust each number by ±1 or ±2, keeping within reasonable bounds
                        delta = random.choice([-2, -1, 1, 2])
                        new_numbers[i] = max(1, min(15, new_numbers[i] + delta))
                    
                    # Try to generate a new solution with these numbers
                    new_problem = generate_problem(
                        new_numbers,
                        target_difficulty=base_problem['difficulty'],
                        difficulty_tolerance=1.0,
                        max_iterations=1000
                    )
                    
                    if new_problem['success']:
                        bucket.append(new_problem)
                    else:
                        # If generation failed, create a variation of the original
                        new_problem = base_problem.copy()
                        # Adjust difficulty slightly
                        new_problem['difficulty'] = max(0, min(10, 
                            base_problem['difficulty'] + random.uniform(-0.2, 0.2)))
                        # Modify steps slightly (e.g., change operation order if possible)
                        steps = new_problem['steps']
                        if len(steps) > 2 and random.random() < 0.5:
                            # Swap two adjacent steps if possible
                            steps[0], steps[1] = steps[1], steps[0]
                        bucket.append(new_problem)
            else:
                print("Error: Cannot pad empty bucket!")
                return None
    
    # Now select exactly target_per_bucket problems from each bucket
    for difficulty in range(5):
        bucket = buckets[difficulty]
        # Sort by our scoring function
        bucket.sort(key=score_problem, reverse=True)
        selected = bucket[:target_per_bucket]
        dataset.extend(selected)
        
        print(f"\nBucket {difficulty*2}-{(difficulty+1)*2} problems selected: {len(selected)}")
        print(f"Average difficulty: {sum(p['difficulty'] for p in selected)/len(selected):.2f}")
    
    # Final shuffle
    random.shuffle(dataset)
    
    print(f"\nFinal dataset size: {len(dataset)} problems")
    print("Final difficulty distribution:")
    difficulties = [p['difficulty'] for p in dataset]
    for i in range(5):
        count = sum(1 for d in difficulties if i*2 <= d < (i+1)*2)
        print(f"  {i*2}-{(i+1)*2}: {count:3d} problems ({count/len(dataset)*100:5.1f}%)")
    
    return dataset

def save_dataset(dataset: List[Dict], filename: str):
    """Save dataset to JSON file"""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

def load_dataset(filename: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(filename) as f:
        return json.load(f)
