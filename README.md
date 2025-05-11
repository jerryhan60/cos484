# cos484
Submission for COS 484 Natural Language Processing Final Project

## Authors
Jerry Han, Brandon Cho, Huxley Marvit

## Overview
This repository provides code for our group's COS 484 Natural Language Processing Final Project. 

Demo: https://youtu.be/aYnkOHSz_mU

## Environment Setup  
We use Conda for environment management.  
```bash
# create and activate environment
conda env create -f nlp-proj-env.yml
conda activate nlp-proj-env
```

Directory structure:
1. generate.py: core generation algorithm code. You can run python generate.py to test generation given a prompt. 
2. eval.py: evaluation code for running benchmark evals on our generation functions
3. experiment_ablation.py and experiment_reproduce.py are scripts which we ran to get experimental results for reproduction and ablation studies
4. utils.py: utility functions for loading models

Potential bugs/known issues
1. This code was run on Della / Lambda H100s. The paths variables may point to wrong places and you have to make sure to pre download the required models 

Required downloads (from Huggingface), this list is not exhaustive
1. Qwen/Qwen-2.5-1.5B-Instruct
2. GSAI-ML/LLaDA-8B-Base