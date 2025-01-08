"""
Model training and evaluation for 24 point solver
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import json

def format_example(numbers: List[int], steps: List[str]) -> str:
    """Format a single example for the model"""
    numbers_str = ', '.join(map(str, numbers))
    steps_str = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(steps))
    return f"Numbers: {numbers_str}\nSolution steps:\n{steps_str}\n"

def prepare_dataset(examples: List[Tuple[List[int], List[str]]]) -> List[str]:
    """Prepare dataset for model training"""
    return [format_example(nums, steps) for nums, steps in examples]

class TwentyFourPointSolver:
    def __init__(self, model_name: str = "Qwen/Qwen-0.5B"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
    def train(self, train_data: List[str]):
        """Train the model using SFT"""
        # TODO: Implement training logic
        pass
        
    def evaluate(self, test_data: List[Tuple[List[int], List[str]]]) -> float:
        """Evaluate model accuracy on test set"""
        # TODO: Implement evaluation logic
        return 0.0
