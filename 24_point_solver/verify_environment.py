import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import os

def verify_environment():
    """Verify the training environment setup"""
    print('\nSystem Information:')
    print('-' * 50)
    print(f'CPU cores: {psutil.cpu_count()}')
    memory = psutil.virtual_memory()
    print(f'RAM Total: {memory.total / (1024**3):.1f} GB')
    print(f'RAM Available: {memory.available / (1024**3):.1f} GB')
    
    print('\nPython Environment:')
    print('-' * 50)
    print(f'Python version: {torch.__version__}')
    print(f'Transformers version: {transformers.__version__}')
    
    print('\nGPU Information:')
    print('-' * 50)
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'CUDA version: {torch.version.cuda}')
        props = torch.cuda.get_device_properties(0)
        print(f'GPU memory: {props.total_memory / 1e9:.1f} GB')
        print(f'GPU compute capability: {props.major}.{props.minor}')
    
    # Test model loading
    print('\nTesting Qwen-0.5B Model:')
    print('-' * 50)
    try:
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen-0.5B',
            trust_remote_code=True
        )
        print('Tokenizer loaded successfully')
        
        print('\nLoading model...')
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen-0.5B',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        print('Model loaded successfully')
        
        # Test basic inference
        print('\nTesting inference...')
        input_text = "Calculate 24: Given numbers [3, 8, 3, 8], find operations to make 24."
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('\nTest generation successful!')
        print(f'Input: {input_text}')
        print(f'Output: {output_text}')
        
        return True
    except Exception as e:
        print(f'\nError during model verification: {str(e)}')
        return False

if __name__ == '__main__':
    success = verify_environment()
    print('\nEnvironment verification', 'successful!' if success else 'failed!')
    exit(0 if success else 1)
