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
    # Operation weights
    op_weights = {'+': 1.0, '-': 1.2, '*': 1.5, '/': 2.0}
    
    # Base difficulty from operations
    op_difficulty = sum(op_weights[step.split()[1]] for step in steps if "Apply" in step)
    
    # Number complexity (larger numbers are harder)
    num_difficulty = sum(min(float(n), 13.0) / 2.0 for n in initial_numbers) / len(initial_numbers)
    
    # Solution length factor (including backtracking)
    length_factor = len([step for step in steps if "Apply" in step]) / 3.0
    
    # Operation diversity bonus
    unique_ops = len({step.split()[1] for step in steps if "Apply" in step})
    diversity_bonus = unique_ops * 0.5
    
    # Backtracking penalty
    backtrack_count = len([s for s in steps if "Backtrack" in s])
    backtrack_factor = 1.0 + (backtrack_count * 0.2)  # 20% increase per backtrack
    
    # Calculate final difficulty score (0-10 scale)
    difficulty = (op_difficulty * 0.3 + 
                 num_difficulty * 0.2 + 
                 length_factor * 0.2 + 
                 diversity_bonus * 0.1 + 
                 backtrack_factor * 0.2) * 2
    
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
                  timeout: float = 10.0,  # Increased timeout
                  verbose: bool = True) -> Optional[Solution]:
    """
    Solve 24-point problem using A* search with detailed exploration.
    Returns solution steps with comprehensive backtracking information.
    """
    import time
    start_time = time.time()
    
    initial_state = State(
        numbers=[float(n) for n in numbers],
        operations=[
            f"Initial state:",
            f"  Numbers available: {numbers}",
            f"  Target value: {target}",
            f"  Starting heuristic calculation..."
        ],
        target=target
    )
    
    # Priority queue for A* search
    frontier = PriorityQueue()
    initial_h = calculate_heuristic(initial_state.numbers, target)
    initial_state.operations.append(
        f"  Initial heuristic value: {initial_h:.2f}\n"
        f"Beginning A* search with initial state..."
    )
    frontier.put((initial_h, 0, initial_state))
    
    # Track visited states and their parents for backtracking
    visited = {}  # state_key -> (parent_key, operation, min_h, step_count)
    initial_key = tuple(sorted(initial_state.numbers))
    visited[initial_key] = (None, None, initial_h, 0)
    
    # Track explored states and paths for detailed search process
    explored_states = []
    exploration_paths = []
    
    # Counter for tiebreaking and progress tracking
    counter = 1
    states_explored = 0
    max_states = 2000  # Increased state limit
    
    while not frontier.empty() and states_explored < max_states:
        # Check timeout
        if time.time() - start_time > timeout:
            if verbose:
                print(f"Search timed out after {timeout} seconds")
            return None
            
        h_value, _, current_state = frontier.get()
        current_key = tuple(sorted(current_state.numbers))
        states_explored += 1
        
        # Record detailed exploration
        explored_states.append((current_key, h_value))
        step_count = len(current_state.operations)
        
        # Add detailed exploration steps
        current_state.operations.extend([
            f"\nExploration Step {states_explored}:",
            f"  Current numbers: {[float(n) for n in current_state.numbers]}",
            f"  Heuristic value: {h_value:.2f}",
            f"  Distance to target: {abs(sum(current_state.numbers) - target):.2f}",
            f"  States explored so far: {states_explored}",
            "  Analyzing possible operations..."
        ])
        
        # Track exploration path
        if len(current_state.numbers) < len(numbers):
            parent_key = visited[current_key][0]
            if parent_key:
                exploration_paths.append((parent_key, current_key))
        
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
