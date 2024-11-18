# Model Comparison Code

This folder contains everything needed to perform model comparisons.

## Setup

1. Transfer this folder to another machine using `scp` (if needed):

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

2. Run the script 

```bash
    ./run_model_comparison.sh
```

3. results stored in saved_models and evaluation_results.csv, feature_importance_results.csv and various model results. 

