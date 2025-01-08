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
    
    # Add dependency installation cell
    install_code = """\
!pip install numpy tqdm matplotlib
print("Dependencies installed successfully!")"""
    nb.cells.append(nbf.v4.new_code_cell(install_code))
    
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
