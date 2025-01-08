"""
Script to run the 24 point solver training in Google Colab
"""
import os
from data_generator import generate_dataset, save_dataset
from model import TwentyFourPointSolver, prepare_dataset

def main():
    # Generate datasets
    print("Generating training dataset...")
    train_data = generate_dataset(num_samples=1000)
    save_dataset(train_data, "train_dataset.json")
    
    print("Generating test dataset...")
    test_data = generate_dataset(num_samples=100)
    save_dataset(test_data, "test_dataset.json")
    
    # Initialize and train model
    print("Initializing model...")
    solver = TwentyFourPointSolver()
    
    print("Preparing training data...")
    train_examples = prepare_dataset(train_data)
    
    print("Training model...")
    solver.train(train_examples)
    
    print("Evaluating model...")
    accuracy = solver.evaluate(test_data)
    print(f"Test accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # Instructions for Colab:
    print("""
    To use this in Colab:
    1. Clone the repository:
       !git clone https://github.com/YOUR_REPO/24_point_solver.git
    2. Install the package:
       !cd 24_point_solver && pip install -e .
    3. Run this script:
       !python colab_runner.py
    """)
    main()
