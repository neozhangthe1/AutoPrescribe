from setuptools import setup, find_packages

setup(
    name="24_point_solver",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.21.0",
    ],
    author="Devin",
    description="24 Point Game Solver using LLM with Long Chain-of-Thought",
    python_requires=">=3.8",
)
