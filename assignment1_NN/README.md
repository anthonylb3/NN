# UvA Neural Networks 1 Assignments, 2025-2026 edition

## Assignment 1

You can find all necessary instructions in the PDF on Canvas.

We provide to you simple unit tests that can check your implementation. However, be aware that even if all tests are passed, it still doesn't mean that your implementation is fully correct. You can find tests in **unittests.py**.

We also provide a Conda environment you can use to install the necessary Python packages. A .yml file is provided for installing the necessary environments. This is done using Miniconda installed through the following link: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

To install the environment run: 

```bash
# 1. Create a virtual environment
conda env create -f dl2025_cpu.yml

# 2. Activate the environment
conda activate dl2025
```

If you have a GPU available you can also install the environment using the .yml file ending on "_gpu". For MacOS use the 'mps' as a device instead of CUDA for utilizing the GPU. 
