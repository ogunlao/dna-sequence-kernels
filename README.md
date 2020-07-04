# Predict whether DNA sequence region is binding site using Kernels

> Aim of the project is to predict whether a DNA sequence region is binding site to a specific transcription factor. Only kernel methods are allowed.

The project was done via a Kaggle competition in which my team implemented various kernels for large dimensional data.

## Getting Started

The `MAIN_SCRIPT` file is the starting point in the code. It imports other functions and build the kernels before use.

Clone the repository using the following command;

```shell
git clone https://github.com/ogunlao/dna-sequence-kernels.git
```

```shell
cd dna-sequence-kernels
```

then run the python command in your terminal;

```python
python MAIN_SCRIPT.py
```

This builds the kernels used and run the code for predictions. On your first run, you may need to wait for some minutes to build the kernel. Subsequent runs will be faster.

## Overview

Spectrum kernel and Weighted degree kernel were used for this project. The weighted degree kernel gave better results as seen in our reports. A `PRESENTATION.pdf` file can be found in the repository which summarizes our findings during the project.

Algorithms

- Kernel Logistic Regression
- Kernel Ridge Regression
- Kernel SVM

## Challenges

- The number of examples in the dataset was small which could give room for overfitting. We countered this by cross-validating our experiments.
- Also, kernels work in a large dimensional space which implies that regularization is a very important aspect of the algorithm.

Enjoy.