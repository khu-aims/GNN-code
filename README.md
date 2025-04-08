# GNN-code

## Requirements

This project uses the Python packages listed in `requirements.txt`. Please install them before running experiments.

## Data Preparation (Skip if you already have the data)

1. **Generate MSN Dataset**  
   Create MSN data such that the variable name is `"connectivity"`, and save it as `corr_{subject name}.mat`.  
   Place these files in the `data/SZ_MSN` directory.  
   For instructions on generating the MSN data, refer to `data/SZ_MSN/MSNCconstruction.ipynb`.

2. **Create phenotype.csv**  
   Make sure the class label column is named `"DX_GROUP"`.

3. **Create subject_IDs.txt**  
   List the filenames of the subjects to be used for experiments.

## Configuration

4. **Set the Root Folder**  
   In `Parser.py`, configure the `root_folder` variable to point to your data directory according to your environment.

## Running the Code

5. **Run the Main Script**  
   Execute the following in your terminal:
   ```bash
   python main.py
   ```

6. **Adjust Hyperparameters**  
   You can modify various hyperparameters based on `opt.py`. For example:
   ```bash
   python main.py --lr 0.001 --msn_threshold 0.3 --encoder EA --hgc 128
   ```

## Model Comparison

7. The following scripts are used for model comparison experiments:
   - `main_rf.py`: Random Forest
   - `main_svm.py`: Support Vector Machine
   - `main_knn.py`: K-Nearest Neighbors
   - `main_dnn.py`: Deep Neural Network
   - `main_dnn_batch.py`: DNN with batch processing  
     *(The main difference between `main_dnn.py` and `main_dnn_batch.py` is the use of batch processing. Batch processing is recommended.)*

![image](https://zenodo.org/badge/DOI/10.5281/zenodo.15172055.svg)
