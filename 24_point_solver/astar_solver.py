from typing import List, Tuple, Set, Dict, Optional, NamedTuple
from dataclasses import dataclass
from queue import PriorityQueue
import math
import random
from collections import defaultdict

@dataclass
class State:
    """Represents a state in the search space"""
    numbers: List[float]  # Current numbers available
    operations: List[str]  # Operations performed to reach this state
    target: float = 24.0  # Target number to reach
    
    def __lt__(self, other):
        """Required for PriorityQueue"""
        return len(self.numbers) < len(other.numbers)

class Solution(NamedTuple):
    """Represents a complete solution"""
    steps: List[str]  # List of operations performed
    final_value: float  # Final value reached
    difficulty: float  # Calculated difficulty of the solution

def calculate_heuristic(numbers: List[float], target: float = 24.0) -> float:
    """
    Calculate heuristic value for A* search.
    Lower value means closer to solution.
    """
    if len(numbers) == 1:
        return abs(numbers[0] - target)
    
    # For multiple numbers, use the difference between their sum/product and target
    # as a simple heuristic
    sum_diff = abs(sum(numbers) - target)
    prod = math.prod(numbers)
    prod_diff = abs(prod - target)
    
    return min(sum_diff, prod_diff)

def calculate_difficulty(steps: List[str], initial_numbers: List[int]) -> float:
    """Calculate solution difficulty based on operations and numbers"""
    # Operation weights (division and subtraction are harder)
    op_weights = {'+': 1.0, '-': 1.5, '*': 1.2, '/': 2.0}
    
    # Count operations
    op_counts = {op: 0 for op in op_weights}
    for step in steps:
        if "Apply" in step:
            for op in op_weights:
                if f" {op} " in step:
                    op_counts[op] += 1
    
    # Base difficulty from operations (weighted sum)
    op_difficulty = sum(op_weights[op] * count for op, count in op_counts.items())
    
    # Number complexity factors
    numbers = [float(n) for n in initial_numbers]
    num_difficulty = (
        sum(min(n, 13.0) / 2.0 for n in numbers) / len(numbers) +  # Size factor
        (max(numbers) - min(numbers)) / 13.0 +  # Range factor
        (len(set(numbers)) / len(numbers))  # Uniqueness factor
    )
    
    # Solution complexity factors
    total_ops = sum(op_counts.values())
    unique_ops = len([op for op, count in op_counts.items() if count > 0])
    backtrack_count = len([s for s in steps if "Backtrack" in s])
    
    # Calculate raw component scores
    op_weights = {'+': 1.0, '-': 1.5, '*': 1.2, '/': 2.0}
    op_counts = defaultdict(int)
    for step in steps:
        if "Apply" in step:
            for op in op_weights:
                if f" {op} " in step:
                    op_counts[op] += 1
    
    # Operation complexity (0-3 points)
    op_score = sum(op_weights[op] * count for op, count in op_counts.items()) / 4.0
    
    # Number complexity (0-2 points)
    numbers = [float(n) for n in initial_numbers]
    num_score = (
        sum(1 for n in numbers if n > 10) * 0.5 +  # Large numbers
        (len(set(numbers)) / len(numbers)) * 0.5 +  # Uniqueness
        (max(numbers) - min(numbers)) / 13.0        # Range
    )
    
    # Solution length (0-2 points)
    total_ops = sum(op_counts.values())
    length_score = total_ops / 3.0
    
    # Operation diversity (0-1.5 points)
    unique_ops = len([op for op, count in op_counts.items() if count > 0])
    diversity_score = (unique_ops / 4.0) * 1.5
    
    # Backtracking complexity (0-1.5 points)
    backtrack_steps = len([s for s in steps if "Backtrack" in s])
    backtrack_score = min(1.5, backtrack_steps / 3.0)
    
    # Calculate base difficulty (0-7 points)
    base_difficulty = (
        min(3.0, op_score) +      # Operations (0-3)
        min(2.0, num_score) +     # Numbers (0-2)
        min(2.0, length_score)    # Length (0-2)
    )
    
    # Apply modifiers
    difficulty = (
        base_difficulty +
        diversity_score +    # Diversity bonus (0-1.5)
        backtrack_score     # Backtracking complexity (0-1.5)
    )
    
    # Scale to 1-10 range with sigmoid-like curve
    # This creates more variation in the middle range
    normalized = difficulty / 10.0  # Scale to 0-1
    sigmoid = 1.0 / (1.0 + math.exp(-6 * (normalized - 0.5)))  # Steeper sigmoid
    scaled_difficulty = 1.0 + sigmoid * 9.0  # Map to 1-10 range
    
    # Ensure minimum difficulty of 1.0 and maximum of 10.0
    return min(10.0, max(1.0, difficulty))

def apply_operation(a: float, b: float, op: str) -> Optional[float]:
    """Apply arithmetic operation and return result if valid"""
    try:
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/' and abs(b) > 1e-10:
            return a / b
        return None
    except:
        return None

def get_next_states(current: State) -> List[Tuple[State, float]]:
    """Generate all possible next states from current state"""
    next_states = []
    numbers = current.numbers
    
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            a, b = numbers[i], numbers[j]
            remaining = numbers[:i] + numbers[i+1:j] + numbers[j+1:]
            
            for op in ['+', '-', '*', '/']:
                # Try operation in both orders for non-commutative ops
                results = [(a, b, op)]
                if op in ['-', '/']:
                    results.append((b, a, op))
                
                for x, y, operation in results:
                    result = apply_operation(x, y, operation)
                    if result is not None:
                        new_numbers = remaining + [result]
                        step = f"Apply {operation} to {x} and {y} to get {result}"
                        new_state = State(
                            numbers=new_numbers,
                            operations=current.operations + [step],
                            target=current.target
                        )
                        h_value = calculate_heuristic(new_numbers, current.target)
                        next_states.append((new_state, h_value))
    
    return next_states

def solve_24_astar(numbers: List[int], target: float = 24.0, 
                  tolerance: float = 1e-10,
                  timeout: float = 15.0,  # Increased timeout further
                  verbose: bool = True) -> Optional[Solution]:
    """
    Solve 24-point problem using A* search with comprehensive exploration tracking.
    Returns detailed solution steps with extensive backtracking information.
    """
    import time
    import math
    from functools import reduce
    from operator import mul
    
    start_time = time.time()
    best_h_value = float('inf')
    last_state = None
    
    def prod(numbers):
        return reduce(mul, numbers, 1)
    
    # Initialize with detailed problem description
    initial_state = State(
        numbers=[float(n) for n in numbers],
        operations=[
            "Problem Analysis:",
            "-------------",
            f"Goal: Find a sequence of arithmetic operations to reach {target}",
            f"Available numbers: {numbers}",
            "",
            "Initial State Analysis:",
            "-------------------",
            "1. Number Properties:",
            f"   - Count: {len(numbers)} numbers",
            f"   - Sum: {sum(numbers)}",
            f"   - Product: {prod(numbers)}",
            f"   - Max: {max(numbers)}",
            f"   - Min: {min(numbers)}",
            "",
            "2. Search Strategy:",
            "   - Using A* search algorithm",
            "   - Heuristic: Minimum difference to target",
            "   - Operations allowed: +, -, *, /",
            "",
            "3. Search Space Analysis:",
            f"   - Maximum possible states: {len(numbers) * (len(numbers) - 1) * 4}",
            f"   - Maximum search depth: {len(numbers) - 1}",
            f"   - Branching factor: {len(numbers) * (len(numbers) - 1) * 2}",
            "",
            "4. Solution Requirements:",
            f"   - Target value: {target}",
            f"   - Tolerance: {tolerance}",
            f"   - Time limit: {timeout} seconds",
            "",
            "Starting Search Process:",
            "--------------------",
            "Search Log:",
            "----------"
        ],
        target=target
    )
    
    # Priority queue for A* search with detailed tracking
    frontier = PriorityQueue()
    initial_h = calculate_heuristic(initial_state.numbers, target)
    initial_state.operations.extend([
        "Initial State Evaluation:",
        f"  Numbers: {[float(n) for n in initial_state.numbers]}",
        f"  Heuristic value: {initial_h:.2f}",
        f"  Distance to target: {abs(sum(initial_state.numbers) - target):.2f}",
        "",
        "Search Log:",
        "----------"
    ])
    frontier.put((initial_h, 0, initial_state))
    
    # Enhanced state tracking
    visited = {}  # state_key -> (parent_key, operation, min_h, step_count, path)
    initial_key = tuple(sorted(initial_state.numbers))
    visited[initial_key] = (None, None, initial_h, 0, [initial_key])
    
    # Comprehensive search process tracking
    explored_states = []
    exploration_paths = []
    dead_ends = []  # Track states that led to no solution
    
    # Search control
    counter = 1
    states_explored = 0
    max_states = 3000  # Further increased state limit
    backtrack_count = 0
    
    while not frontier.empty() and states_explored < max_states:
        # Check timeout with detailed message
        if time.time() - start_time > timeout:
            if verbose:
                print(f"Search timed out after {timeout} seconds")
            if last_state:
                last_state.operations.extend([
                    "",
                    "Search Terminated:",
                    f"- Time limit ({timeout}s) exceeded",
                    f"- States explored: {states_explored}",
                    f"- Current best heuristic: {best_h_value:.2f}",
                    f"- Best state reached: {[float(n) for n in last_state.numbers]}"
                ])
            return None
            
        h_value, _, current_state = frontier.get()
        current_key = tuple(sorted(current_state.numbers))
        states_explored += 1
        
        # Update best state tracking
        if h_value < best_h_value:
            best_h_value = h_value
            last_state = current_state
        
        # Enhanced state exploration logging
        explored_states.append((current_key, h_value))
        step_count = len(current_state.operations)
        
        # Get path to current state
        current_path = []
        temp_key = current_key
        while temp_key in visited:
            current_path.append(temp_key)
            parent_info = visited[temp_key]
            if parent_info[0] is None:
                break
            temp_key = parent_info[0]
        current_path.reverse()
        
        # Get next possible states for analysis
        next_states = []
        for i in range(len(current_state.numbers)):
            for j in range(i + 1, len(current_state.numbers)):
                for op in ['+', '-', '*', '/']:
                    a, b = current_state.numbers[i], current_state.numbers[j]
                    result = apply_operation(a, b, op)
                    if result is not None:
                        # Create new state with the result
                        new_numbers = current_state.numbers.copy()
                        new_numbers.pop(j)
                        new_numbers[i] = result
                        
                        # Generate detailed operation description
                        op_desc = (
                            f"Apply {op} to {a:.2f} and {b:.2f} to get {result:.2f}\n"
                            f"  - Operation: {a:.2f} {op} {b:.2f} = {result:.2f}\n"
                            f"  - Remaining numbers: {[float(n) for n in new_numbers]}\n"
                            f"  - Progress: {len(new_numbers)} numbers remaining\n"
                            f"  - Target distance: {abs(sum(new_numbers) - target):.2f}"
                        )
                        
                        next_h = calculate_heuristic(new_numbers, target)
                        next_states.append((new_numbers, op_desc, next_h))
        
        # Sort next states by heuristic value
        next_states.sort(key=lambda x: x[2])
        
        # Detailed exploration log with next state analysis
        current_state.operations.extend([
            "",
            f"Exploration Step {states_explored}:",
            "-------------------",
            "1. Current State Analysis:",
            f"   Numbers: {[float(n) for n in current_state.numbers]}",
            f"   Heuristic value: {h_value:.2f}",
            f"   Distance to target: {abs(sum(current_state.numbers) - target):.2f}",
            f"   Best heuristic so far: {best_h_value:.2f}",
            "",
            "2. Search Progress:",
            f"   States explored: {states_explored}",
            f"   Current depth: {len(numbers) - len(current_state.numbers)}",
            f"   Backtrack count: {backtrack_count}",
            f"   Queue size: {frontier.qsize()}",
            "",
            "3. Path Analysis:",
            "   Current path:",
            "   " + " -> ".join(str(list(p)) for p in current_path),
            "",
            "4. Next State Analysis:",
            "   Possible next states (top 5):"
        ])
        
        # Add extremely detailed next state analysis with mathematical properties
        for i, (next_nums, op_desc, next_h) in enumerate(next_states[:5]):
            op_type = op_desc.split()[1]
            operands = [float(op_desc.split()[3]), float(op_desc.split()[5])]
            result = float(op_desc.split()[-1])
            
            current_state.operations.extend([
                f"\n   {i+1}. Comprehensive Next State Analysis:",
                "   ==============================",
                f"   Operation Details:",
                f"{op_desc}",
                f"   Mathematical Properties:",
                f"      1. Operation Characteristics:",
                f"         - Type: {op_type}",
                f"         - Properties:",
                f"           * Commutative: {op_type in ['+', '*']}",
                f"           * Associative: {op_type in ['+', '*']}",
                f"           * Has inverse: {op_type in ['+', '*']}",
                f"           * Identity element: {1 if op_type == '*' else 0}",
                f"      2. Number Properties:",
                f"         - Operands: {operands}",
                f"           * Sum: {sum(operands):.2f}",
                f"           * Product: {math.prod(operands):.2f}",
                f"           * Ratio: {f'{operands[0]/operands[1]:.2f}' if abs(operands[1]) > 1e-10 else 'undefined'}",
                f"           * GCD: {str(math.gcd(int(operands[0]), int(operands[1]))) if all(x.is_integer() and abs(x) < 1e6 for x in operands) else 'N/A'}",
                f"         - Result: {f'{result:.2f}' if not math.isinf(result) and not math.isnan(result) else 'undefined'}",
                f"           * Integer?: {'Yes' if not math.isinf(result) and not math.isnan(result) and result.is_integer() else 'No' if not math.isinf(result) and not math.isnan(result) else 'N/A'}",
                f"           * Sign: {'+'  if not math.isinf(result) and not math.isnan(result) and result > 0 else '-' if not math.isinf(result) and not math.isnan(result) and result < 0 else '0' if not math.isinf(result) and not math.isnan(result) and result == 0 else 'undefined'}",
                f"   State Evaluation:",
                f"      1. Heuristic Analysis:",
                f"         - New value: {next_h:.2f}",
                f"         - Previous: {h_value:.2f}",
                f"         - Change: {h_value - next_h:+.2f}",
                f"         - Relative improvement: {f'{((h_value - next_h)/h_value*100):.1f}' if abs(h_value) > 1e-10 else '0.0'}%",
                f"      2. Target Analysis:",
                f"         - Current distance: {abs(sum(next_nums) - target):.2f}",
                f"         - Progress: {f'{(1 - min(1, abs(sum(next_nums) - target)/target))*100:.1f}'}% to goal",
                f"   Numerical Analysis:",
                f"      1. State Properties:",
                f"         - Numbers: {next_nums}",
                f"         - Count: {len(next_nums)}",
                f"         - Unique values: {len(set(next_nums))}",
                f"      2. Statistical Measures:",
                f"         - Sum: {sum(next_nums):.2f}",
                f"         - Product: {math.prod(next_nums):.2f}",
                f"         - Mean: {sum(next_nums)/len(next_nums):.2f}",
                f"         - Range: {max(next_nums) - min(next_nums):.2f}",
                f"      3. Distribution:",
                f"         - Minimum: {min(next_nums):.2f}",
                f"         - Maximum: {max(next_nums):.2f}",
                f"         - Median: {sorted(next_nums)[len(next_nums)//2]:.2f}",
                f"   Search Progress:",
                f"      1. Path Metrics:",
                f"         - Depth: {4-len(next_nums)}",
                f"         - Branching factor: {len(next_states)}",
                f"         - Path length: {len(current_state.operations)}",
                f"      2. Completion Analysis:",
                f"         - Progress: {((4-len(next_nums))/3*100):.1f}%",
                f"         - Steps taken: {3-len(next_nums)+1}",
                f"         - Steps remaining: {len(next_nums)-1}",
                "\n"
            ])
            
        current_state.operations.extend([
            "",
            "5. Decision Process:",
            f"   Total next states: {len(next_states)}",
            f"   Best next heuristic: {next_states[0][2] if next_states else 'N/A'}",
            f"   Worst next heuristic: {next_states[-1][2] if next_states else 'N/A'}",
            "   Selected operation: Choosing best available next state",
            "",
            "6. Memory Analysis:",
            f"   Visited states: {len(visited)}",
            f"   Frontier size: {frontier.qsize()}",
            f"   Exploration history: {len(explored_states)} states"
        ])
        
        # Track exploration with backtracking
        if len(current_state.numbers) < len(numbers):
            parent_key = visited[current_key][0]
            if parent_key:
                exploration_paths.append((parent_key, current_key))
                # Check if we're backtracking
                if parent_key in [p[1] for p in exploration_paths[-5:]]:
                    backtrack_count += 1
                    current_state.operations.extend([
                        "",
                        "Backtracking Detected:",
                        f"   Previous state: {list(parent_key)}",
                        f"   Current state: {list(current_key)}",
                        f"   Reason: Better path found",
                        ""
                    ])
        
        # Check if we've reached the target
        if (len(current_state.numbers) == 1 and 
            abs(current_state.numbers[0] - target) < tolerance):
            
            # Build solution path (limit backtracking info)
            solution_steps = current_state.operations.copy()
            
            # Add limited backtracking information
            backtrack_count = 0
            for i, (state_key, h) in enumerate(explored_states[:50]):  # Limit to first 50
                if i > 0:
                    parent_key, op, _ = visited[state_key]
                    if parent_key:
                        if op and "Apply" in op:
                            solution_steps.append(op)
                        else:
                            backtrack_count += 1
                            solution_steps.append(
                                f"Backtrack to previous state with numbers "
                                f"{[float(n) for n in parent_key]}"
                            )
            
            return Solution(
                steps=solution_steps,
                final_value=current_state.numbers[0],
                difficulty=calculate_difficulty(
                    solution_steps,
                    numbers
                )
            )
        
        # Generate and evaluate next states
        next_states = get_next_states(current_state)
        
        # Process next states (limit branching)
        for next_state, next_h in sorted(next_states, key=lambda x: x[1])[:10]:  # Limit branching
            state_key = tuple(sorted(next_state.numbers))
            if state_key not in visited:
                visited[state_key] = (current_key, next_state.operations[-1], next_h)
                frontier.put((next_h, counter, next_state))
                counter += 1
            elif next_h < visited[state_key][2]:  # Found a better path
                visited[state_key] = (current_key, next_state.operations[-1], next_h)
                frontier.put((next_h, counter, next_state))
                counter += 1
                if len(current_state.operations) < 100:  # Limit operation history
                    current_state.operations.append(
                        f"Found better path to state {[float(n) for n in next_state.numbers]}"
                    )
    
    return None

def generate_problems(num_problems: int = 100,
                     min_difficulty: float = 1.0,
                     max_difficulty: float = 10.0,
                     target: float = 24.0) -> List[Dict]:
    """
    Generate problems with solutions using A* search.
    Returns list of problems with detailed solutions.
    """
    problems = []
    difficulty_buckets = defaultdict(int)
    bucket_size = (max_difficulty - min_difficulty) / 5
    
    while len(problems) < num_problems:
        # Generate random numbers
        numbers = [random.randint(1, 13) for _ in range(4)]
        
        # Try to solve with A*
        solution = solve_24_astar(numbers, target)
        
        if solution:
            # Calculate difficulty bucket
            bucket = int((solution.difficulty - min_difficulty) / bucket_size)
            
            # Only add if bucket isn't too full
            if difficulty_buckets[bucket] < (num_problems / 5) * 1.2:
                problem = {
                    'numbers': numbers,
                    'target': target,
                    'steps': solution.steps,
                    'final_value': solution.final_value,
                    'difficulty': solution.difficulty,
                    'success': True
                }
                problems.append(problem)
                difficulty_buckets[bucket] += 1
    
    return problems

if __name__ == '__main__':
    # Example usage
    test_numbers = [4, 7, 8, 8]
    solution = solve_24_astar(test_numbers)
    
    if solution:
        print(f"\nSolution found for {test_numbers}:")
        print(f"Target: 24")
        print(f"Difficulty: {solution.difficulty:.2f}")
        print("\nSteps:")
        for step in solution.steps:
            print(f"  {step}")
        print(f"Final value: {solution.final_value}")
    else:
        print(f"\nNo solution found for {test_numbers}")
    
    # Generate and analyze problems
    print("\nGenerating balanced problem set...")
    problems = generate_problems(100)
    
    # Analyze difficulty distribution
    difficulties = [p['difficulty'] for p in problems]
    print(f"\nGenerated {len(problems)} problems")
    print(f"Average difficulty: {sum(difficulties)/len(difficulties):.2f}")
    print(f"Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}")
    
    # Show difficulty distribution
    buckets = defaultdict(int)
    for d in difficulties:
        bucket = int(d * 2) / 2  # 0.5 size buckets
        buckets[bucket] += 1
    
    print("\nDifficulty distribution:")
    for bucket in sorted(buckets.keys()):
        count = buckets[bucket]
        print(f"{bucket:4.1f}-{bucket+0.5:4.1f}: {count:3d} problems "
              f"({count/len(problems)*100:5.1f}%)")
