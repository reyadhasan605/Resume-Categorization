# Resume Categorization


## Objective: 

Create, implement, and train a machine learning model to automatically categorize resumes by domain (e.g., sales, marketing, etc.). Next, create a script that can be executed from the command line to process a batch of resumes, categorize them, and output the results to both directory structures and a CSV file.

## Files and directories

### Notebooks

data_exploration_and_processing_ml_models.ipynb contains the data vizualization, preprocessing and all the script for training machine learning models.

Best_model.py contains the BERT model which performs best in our case. It contains training script for BERT model.
      
      
### Script file
```
script.py
```
it contains script to run the model for categorization resume. if you want to run or test mac hine learning model then you have to run script_for_ml_models.py"
### environment file
```
environment.yml
```
It has the necessary libraries along with its versions


## Folders

### dataset

It contains the dataset for training the model

### model

It contains the exported model including machine learning models, tfidf_vector model and also Bert_model. You can download the model from: https://drive.google.com/drive/folders/1gfKlY7yUT8xoU_Z-9A3gjm-JYn9pwGQY?usp=sharing

### input_cv
it contains the test resumes for sorting  in appropriate folder

### OUTPUT

It contains 2 things:
                    1. All the sorted resume folders
                    2. The required categorized_resumes.csv file

### documentation

It contains the 1. Model Selection.md 2. Text Preprocessing.md files for the complete documentation of the project Resume Categorization


## Environment Setup

1. Install [Anaconda](https://www.anaconda.com/download/)
2. Install the environment using following command
```
conda env create -f environment.yml
```
3. Activate the environment
```
conda activate resume_categorization
```

## Instructions to run

1. Open command prompt/terminal/anaconda prompt
2. Goto the directory: `cd C:\Users\{user}\{your directory}`
3. Run script: `python script.py --input_dir "your_input_dir" --output_dir "your_ouput_dir"`
4. output will be saved on "./OUTPUT"
