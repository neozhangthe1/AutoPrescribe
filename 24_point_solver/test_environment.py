"""
Test script to verify the environment setup
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from search import solve_24_point
from data_generator import generate_dataset
import os

def test_environment():
    # Test 1: Check CUDA availability
    print("Testing CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test 2: Check Qwen model loading
    print("\nTesting Qwen model loading...")
    try:
        from huggingface_hub import model_info, HfApi
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Using the smallest Qwen model as specified
        print(f"Attempting to load model: {model_name}")
        
        # Get model info and verify it exists
        info = model_info(model_name)
        print(f"Model found: {info.modelId}")
        print(f"License: {[tag for tag in info.tags if 'license:' in tag]}")
        print(f"Downloads: {info.downloads}")
        
        # Accept the license if needed
        api = HfApi()
        try:
            api.agree_to_license(model_name)
            print("License accepted successfully")
        except Exception as e:
            print(f"Note: License agreement failed or not needed: {str(e)}")
        
        print("Loading tokenizer and model...")
        
        # Load tokenizer with proper configuration
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            pad_token="<|extra_0|>"
        )
        print("✓ Tokenizer loaded successfully")
        
        # Configure model loading with proper settings
        print("Loading model (this may take a few minutes)...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "load_in_8bit": True if torch.cuda.is_available() else False,
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        print("✓ Model loaded successfully")
        
        # Verify model works with a simple test input
        print("\nTesting model with a simple input...")
        test_input = "Calculate 24 using these numbers: 4, 8, 3, 6"
        inputs = tokenizer(test_input, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✓ Model generated response:")
        print(response)
        print("✓ Successfully loaded Qwen model and tokenizer")
    except Exception as e:
        print(f"✗ Failed to load Qwen model: {str(e)}")
        return False
    
    # Test 3: Check 24-point solver
    print("\nTesting 24-point solver...")
    try:
        numbers = [4, 6, 7, 8]
        success, steps = solve_24_point(numbers)
        print(f"✓ Successfully ran solver on {numbers}")
        print(f"Solution found: {success}")
        if success:
            print("Steps:", steps)
    except Exception as e:
        print(f"✗ Failed to run solver: {str(e)}")
        return False
    
    # Test 4: Check data generation
    print("\nTesting data generation...")
    try:
        dataset = generate_dataset(num_samples=2)
        print(f"✓ Successfully generated {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to generate data: {str(e)}")
        return False
    
    print("\nAll environment tests passed successfully!")
    return True

if __name__ == "__main__":
    test_environment()
