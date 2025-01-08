import nbformat as nbf
import re

def create_notebook_from_script(script_path, notebook_path):
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add initial markdown cell with setup instructions
    setup_md = """\
# 24-Point Problem Solver - Dataset Generation

This notebook implements a dataset generator for 24-point arithmetic problems with difficulty balancing. The generator creates problems with controlled difficulty levels and ensures a balanced distribution across different ranges.

## Setup Instructions

1. Run the first cell to install required dependencies
2. Run subsequent cells in order to generate and analyze the dataset
3. The final dataset will be saved as '24_point_dataset.json'

## Features

- Generates unique 24-point arithmetic problems
- Controls problem difficulty (range 0-10)
- Ensures uniform distribution across difficulty levels
- Provides detailed analysis and visualization of results
"""
    nb.cells.append(nbf.v4.new_markdown_cell(setup_md))
    
    # Add environment setup and dependency installation cells
    setup_code = """\
%%capture
# Install required packages
!pip install numpy tqdm matplotlib transformers torch accelerate
!pip install git+https://github.com/QwenLM/Qwen.git
"""
    nb.cells.append(nbf.v4.new_code_cell(setup_code))
    
    # Add environment verification cell
    verify_code = """\
# Check Python version
import sys
print(f"Python version: {sys.version}")

# Install required packages
!pip install --quiet numpy tqdm matplotlib

# Verify installations
import pkg_resources
required_packages = ['numpy', 'tqdm', 'matplotlib']
installed_packages = [pkg.key for pkg in pkg_resources.working_set]
print("\nPackage versions:")
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not found")

# Check for GPU availability (if needed)
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("\nPyTorch not installed (not required for basic operation)")

print("\nBase environment setup completed successfully!")"""
    nb.cells.append(nbf.v4.new_code_cell(verify_code))
    
    # Add model verification cell
    model_code = """\
# Verify model and context length capabilities
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def check_model_and_context():
    print("Checking model and context length capabilities...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)
        print("\\nTokenizer loaded successfully")
        print(f"Maximum context length: {tokenizer.model_max_length}")
        
        # Test context length
        test_text = "test " * 2000  # Create ~4k tokens of text
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Successfully encoded {len(tokens['input_ids'][0])} tokens")
        
        # Load model in 8-bit to save memory
        print("\\nLoading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B",
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True
        )
        print("Model loaded successfully")
        
        # Test model with small input
        input_text = "Calculate 24: Given numbers [2, 8, 9, 1], find operations to make 24."
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\\nTest generation successful!")
        print(f"Input: {input_text}")
        print(f"Output: {response}")
        
        return True
    except Exception as e:
        print(f"\\nError during model verification: {str(e)}")
        return False

# Run verification
model_ok = check_model_and_context()
if model_ok:
    print("\\n✅ Environment fully verified and ready!")
else:
    print("\\n⚠️ Some verifications failed. Check the errors above.")"""
    nb.cells.append(nbf.v4.new_code_cell(model_code))
    
    # Read the script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Split the script into cells based on cell markers
    cells = []
    current_cell = []
    current_type = None
    
    for line in script_content.split('\n'):
        if line.startswith('# %% [markdown]'):
            # Save previous cell if it exists
            if current_cell:
                cell_content = '\n'.join(current_cell)
                if current_type == 'markdown':
                    # Extract markdown content from triple quotes
                    match = re.match(r'"""(.*?)"""', cell_content, re.DOTALL)
                    if match:
                        cell_content = match.group(1).strip()
                    cells.append(nbf.v4.new_markdown_cell(cell_content))
                else:
                    cells.append(nbf.v4.new_code_cell(cell_content))
            current_cell = []
            current_type = 'markdown'
        elif line.startswith('# %% [code]'):
            # Save previous cell if it exists
            if current_cell:
                cell_content = '\n'.join(current_cell)
                if current_type == 'markdown':
                    match = re.match(r'"""(.*?)"""', cell_content, re.DOTALL)
                    if match:
                        cell_content = match.group(1).strip()
                    cells.append(nbf.v4.new_markdown_cell(cell_content))
                else:
                    cells.append(nbf.v4.new_code_cell(cell_content))
            current_cell = []
            current_type = 'code'
        else:
            current_cell.append(line)
    
    # Add the last cell
    if current_cell:
        cell_content = '\n'.join(current_cell)
        if current_type == 'markdown':
            match = re.match(r'"""(.*?)"""', cell_content, re.DOTALL)
            if match:
                cell_content = match.group(1).strip()
            cells.append(nbf.v4.new_markdown_cell(cell_content))
        else:
            cells.append(nbf.v4.new_code_cell(cell_content))
    
    # Add cells to notebook
    nb.cells.extend(cells)
    
    # Add visualization cell
    viz_code = """\
# Visualize difficulty distribution
import matplotlib.pyplot as plt

difficulties = [p['difficulty'] for p in all_problems]

plt.figure(figsize=(12, 6))
plt.hist(difficulties, bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Problem Difficulties')
plt.xlabel('Difficulty')
plt.ylabel('Number of Problems')
plt.grid(True, alpha=0.3)
plt.show()

# Print detailed statistics
print(f"\\nDataset Statistics:")
print(f"Total problems: {len(all_problems)}")
print(f"Average difficulty: {np.mean(difficulties):.2f}")
print(f"Standard deviation: {np.std(difficulties):.2f}")
print(f"Difficulty range: {min(difficulties):.2f} - {max(difficulties):.2f}")

print("\\nDistribution across difficulty ranges:")
for diff_min, diff_max in DIFFICULTY_RANGES:
    count = sum(1 for d in difficulties if diff_min <= d < diff_max)
    print(f"{diff_min:2d}-{diff_max:2d}: {count:3d} problems ({count/len(difficulties)*100:5.1f}%)")

# Save dataset
with open('24_point_dataset.json', 'w') as f:
    json.dump(all_problems, f, indent=2)
print("\\nDataset saved to '24_point_dataset.json'")"""
    nb.cells.append(nbf.v4.new_code_cell(viz_code))
    
    # Set notebook metadata
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.12'
        }
    }
    
    # Write the notebook
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
        print(f"Notebook created successfully at {notebook_path}")
        print("You can now upload this notebook to Google Colab")

if __name__ == '__main__':
    create_notebook_from_script(
        '/home/ubuntu/24_point_solver/colab_24_point_solver.py',
        '/home/ubuntu/24_point_solver/24_point_solver.ipynb'
    )
