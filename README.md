<!-- The code is written with inspiration from Deep Learning in Production Book, adapted for PyTorch -->
# Bijective mapping from sdf to spline 

This code is used for training and  testing the Bijective Mapping Between NURBS Curve and Signed Distance Field


## Requirements

### Install required packages

The training and inference both are performed with CUDA enabled devices

Run

```
conda env create -f environment.yml
conda activate spline
```


## Usage

1. Create a ```lhs_ctrlpts.npy``` which contains the variation of the control points
2. Run  script in ops/generate.py, a images2 file must be there ```python ops/generate.py --iteration_number val --total_iterations val ```.


3. Run training script with ```python main.py $(pwd)/config.txt```. with "mode" set to train



4. Run inference script with ```python main.py $(pwd)/config.txt``  with "mode" set to test

## Validation 
```validation```  folder contains dummy files to test the code for reproducibility

## Visualization of Results
![Result Visualization](test_results_visualization.gif)
