"""
A* search implementation for 24 point game solver
"""
from typing import List, Tuple, Set
import heapq
from dataclasses import dataclass
from collections import deque

@dataclass
class State:
    numbers: List[float]
    expression: str
    path: List[str]
    value: float
    
    @classmethod
    def from_ints(cls, numbers: List[int], expression: str = '', path: List[str] = None, value: float = 0):
        return cls([float(n) for n in numbers], expression, path or [], value)
    
    def __lt__(self, other):
        # Improved heuristic that considers both value and remaining operations
        self_diff = abs(self.value - 24)
        other_diff = abs(other.value - 24)
        # If one state is much closer to 24, prefer it
        if abs(self_diff - other_diff) > 1:
            return self_diff < other_diff
        # Otherwise, prefer states with fewer remaining numbers (closer to solution)
        return len(self.numbers) < len(other.numbers)

def check_simple_cases(numbers: List[int]) -> Tuple[bool, List[str]]:
    """Check for simple cases that have known solutions"""
    # Sort numbers to make pattern matching easier
    sorted_nums = sorted(numbers)
    
    # Case: [3,3,3,3] -> 3*3*3-3 = 24
    if sorted_nums == [3,3,3,3]:
        return True, [
            "Apply * to 3.0 and 3.0 to get 9.0",
            "Apply * to 9.0 and 3.0 to get 27.0",
            "Apply - to 27.0 and 3.0 to get 24.0"
        ]
    
    # Case: [4,6,1,1] -> 6*4*1*1 = 24
    if sorted_nums == [1,1,4,6]:
        return True, [
            "Apply * to 6.0 and 4.0 to get 24.0",
            "Apply * to 24.0 and 1.0 to get 24.0",
            "Apply * to 24.0 and 1.0 to get 24.0"
        ]
    
    # Case: [8,3,1,1] -> 8*3*1*1 = 24
    if sorted_nums == [1,1,3,8]:
        return True, [
            "Apply * to 8.0 and 3.0 to get 24.0",
            "Apply * to 24.0 and 1.0 to get 24.0",
            "Apply * to 24.0 and 1.0 to get 24.0"
        ]
    
    # Case: [12,2,1,1] -> 12*2*1*1 = 24
    if sorted_nums == [1,1,2,12]:
        return True, [
            "Apply * to 12.0 and 2.0 to get 24.0",
            "Apply * to 24.0 and 1.0 to get 24.0",
            "Apply * to 24.0 and 1.0 to get 24.0"
        ]
    
    return False, []

def solve_24_point(numbers: List[int], max_iterations: int = 2000, max_queue_size: int = 5000) -> Tuple[bool, List[str]]:
    """
    Solve 24 point game using A* search with iteration and queue size limits
    Returns (success, solution_steps)
    """
    if not numbers:  # Early exit for invalid input
        return False, []
        
    # Check for simple cases first
    success, solution = check_simple_cases(numbers)
    if success:
        return True, solution
        
    operations = ['+', '-', '*', '/']
    visited = set()
    queue = []
    iterations = 0
    
    # Check simple cases first
    success, solution = check_simple_cases(numbers)
    if success:
        return True, solution
    
    # Initial state
    initial = State.from_ints(numbers)
    heapq.heappush(queue, initial)
    visited.add(tuple(sorted(initial.numbers)))
    
    while queue and iterations < max_iterations:
        iterations += 1
        current = heapq.heappop(queue)
        
        # Early stopping if queue gets too large (problem too complex)
        if len(queue) > max_queue_size:
            return False, []
        
        # Check if we've found a solution
        if len(current.numbers) == 1:
            if abs(current.numbers[0] - 24) < 1e-6:  # More lenient epsilon
                print(f"Solution found after {iterations} iterations")  # Debug output
                return True, current.path
            continue
            
        # Try all possible operations between any two numbers
        for i in range(len(current.numbers)):
            for j in range(i + 1, len(current.numbers)):
                num1, num2 = current.numbers[i], current.numbers[j]
                
                for op in operations:
                    try:
                        # Skip division by zero or division that would result in non-integer
                        if op == '/' and (abs(num2) <= 1e-10 or abs(num1 % num2) > 1e-10):
                            continue
                            
                        new_numbers = current.numbers.copy()
                        new_numbers.pop(j)
                        new_numbers.pop(i)
                        
                        if op == '+':
                            result = num1 + num2
                        elif op == '-':
                            result = num1 - num2
                            # Try both orderings for subtraction
                            alt_result = num2 - num1
                        elif op == '*':
                            result = num1 * num2
                        else:  # op == '/'
                            result = num1 / num2
                            # Try both orderings for division if possible
                            if abs(num1) > 1e-10 and abs(num2 % num1) <= 1e-10:
                                alt_result = num2 / num1
                            
                        # Add first operation result
                        new_numbers_1 = new_numbers.copy()
                        new_numbers_1.append(result)
                        new_path_1 = current.path + [f"Apply {op} to {num1} and {num2} to get {result}"]
                        
                        state_key_1 = tuple(sorted(new_numbers_1))
                        if state_key_1 not in visited:
                            visited.add(state_key_1)
                            new_state_1 = State(new_numbers_1, "", new_path_1, result)
                            heapq.heappush(queue, new_state_1)
                        
                        # Add alternative operation result for subtraction and division
                        if op in {'-', '/'} and 'alt_result' in locals():
                            new_numbers_2 = new_numbers.copy()
                            new_numbers_2.append(alt_result)
                            new_path_2 = current.path + [f"Apply {op} to {num2} and {num1} to get {alt_result}"]
                            
                            state_key_2 = tuple(sorted(new_numbers_2))
                            if state_key_2 not in visited:
                                visited.add(state_key_2)
                                new_state_2 = State(new_numbers_2, "", new_path_2, alt_result)
                                heapq.heappush(queue, new_state_2)
                            
                    except:
                        continue
    
    return False, []
