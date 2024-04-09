## Overview
This project includes several Python scripts that together implement a deep learning model for drug sensitivity prediction using deep learning methods (DNN-PNN). 

## Requirements
Package                  Version
------------------------ -----------
numpy                    1.21.6
pandas                   1.3.5
Pillow                   9.5.0
scikit-learn             1.0.2
scipy                    1.7.3
tensorboard              1.15.0
tensorflow               1.15.0
tensorrt                 8.6.1.post1

## Installation
1. Clone the repository:
```
    git clone [https://github.com/TJU-YangYang//Drug_Sensitivity_Prediction.git]
    cd [DNN-PNN]
```

2. Install required packages:
```
    pip install -r requirements.txt
```

## File description
1. DataLoader_num.py: Manages the loading and preprocessing of numerical data sets.
2. DNN_PNN.py: Implements the Deep Neural Network (DNN) and Product Neural Network (PNN) models.
3. main.py: Serves as the entry point of the program, orchestrating the data loading, model training, and evaluation processes. 
4. metrics.py: Contains functions to calculate various performance metrics for model evaluation. 
5. config.py: Algorithm configuration file, defines the input file, output results folder.


## Usage
To run the code, call the following command:
```
    python main.py
```
