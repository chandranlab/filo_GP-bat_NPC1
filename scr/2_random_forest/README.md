<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Random Forest regressor</h3>
  
  <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/blob/main/img/evaluation.png">
    <img src="/img/evaluation.png" alt="Logo" width="400">
  </a>
  
  <p align="center">
    Training and testing a RF regressor to predict filovirus GP:bat NPC1 binding avidity
    <br />
    <a href="https://github.com/chandranlab/filo_GP-bat_NPC1"><strong>Back to Main»</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/issues">Report Bug</a>
    ·
    <a href="https://github.com/chandranlab/filo_GP-bat_NPC1/issues">Request Feature</a>
  </p>
</div>

<!-- ########################################################################################## -->

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

 Python version: 3.7.6

 Python packages

|Package         | Version  |
|----------------|:--------:|
|sklearn         | 1.0.2    |
|matplotlib      | 3.1.3    |
|numpy           | 1.21.6   |
|pandas          | 1.0.1    |
|seaborn         | 0.11.2   |


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ########################################################################################## -->

<!-- INPUT -->

## Input files

### model[1,2]_params.txt
* Optimized hyper-parameters for the two RF regressors presented in the manuscript (model 1 and model 2)

### model[1,2]_datasets
* model[1,2]\_train: Training dataset with selected features
* model[1,2]\_test: Testing dataset with selected features
* model[1,2]\_llov: Lloviu GP-binding datataset with selected features



<!-- ########################################################################################## -->

<!-- SCRIPT -->

## Script

1. Jupyter notebook: ML.ipynb
    * Class "ML" with basic functions to train and evaluate the RF

2. Jupyter notebook: RF_train_eval.ipynb
    * Loads the hyper-parameters and datasets for each model.
    * Evaluates by 10-fold cross-validation the RF
    * Trains the RF and evaluates it against the testing set and the Llov subset
    * Evaluation output is saved in 'output/model_eval.txt'
    

<p align="right">(<a href="#readme-top">back to top</a>)</p>
