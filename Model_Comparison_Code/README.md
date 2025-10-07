# Model Comparison Code
## Michael Tao and Alex Hoang fall 2024
This folder contains everything needed to perform model comparisons.

## Setup

## 1. Transfer this folder to another machine using `scp` (if needed) and install packages:

```bash
scp -r Model_Comparison_Code <username>@<target_machine>:<destination_path>
cd Model_Comparison_Code
sudo ./setup_environment.sh
```
### What does the above do?
* Update system packages.
* Install required dependencies.
* Ensure GNU grep is updated.
* Prepare the necessary directories

OR 
look in the requirements.txt in order to install the correct packages. 

## 2. Run the script 

```bash
./run_model_comparison.sh
```

To further play with arguments that the script is using(including train_split, val_split, input_steps, output_steps, num_stations, etc.), you can modify run_model_comparison.sh 

## 3. Results
stored in saved_models and evaluation_results.csv, feature_importance_results.csv and various model results. 

### evaluation_results_(shape).csv - main data that is stored
- the name of the csv changes based on the configuration that is being used.
- stores the results of having a single feature be evaluated with the label. The best model is always chosen but not stored. Then the script will run with the top x features until it reaches the maximum amount of features. 
    - only stores MAE since MAE is used to find the best features
- incremental evaluation - based off of the ranking of the best features by MAE, we add these features one at a time to the model to evaluate how it performs. 
- MAE, MAPE, MSE are stored for each model when evaluation type is the incremental type.
- the goal of this is to find the best feature when used solo. Then find the best group of features that work well together.  
- MAE_diff, MSE_diff, MAPE_diff are stored to compare how the model did when it had all the features. Example below:
MAE_diff = MAE_current - original_performance

original_performance was how the model performed with all features. Therefore, the lower and more negative the MAE_diff is, the better that combination of features works. 
 
### feature_importance_results_(shape).csv 
- the name of the csv changes based on the configuration that is being used. 
- stores results of the models after getting rid of one faeature at a time
- stores MSE, MAE, MAPE for each time

## 4. What is the code doing?
This folder in the repository is aimed to train, evaluate, compare various models. It will evaluate the various models based on configurations of input features, input-output steps, and feature importance. These models are known in the code as the original_performance. 

The second half of the project is aimed at finding the best features. This is done by 
1. removing one feature at a time to see how that changed the MSE, MAE, MAPE
2. having solo features with the label to see how each feature affects the  MSE, MAE, MAPE
3. analyzing the top x features to find the best combinations of features. 

The `model_comparison.py` script serves as the main file with the main function. It uses helper functions from `helpers.py` for preprocessing data, defining model architectures, and evaluating performance. The results, including metrics like MAE and MSE, are logged to CSV files.



