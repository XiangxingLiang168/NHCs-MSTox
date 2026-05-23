# NHCs-MSTox

<p align="center">
  <img src="docs/Fig1.jpg" alt="Fig1" width="1200"/>
</p>

## Training of the NHCs-MSTox Model

NHCs-MSTox is a machine learning framework for predicting the toxicity of nitrogen-containing heterocyclic compounds (NHCs). The model was trained using a dataset of 108 NHCs with toxicity values obtained from *Vibrio fischeri* bioluminescence inhibition assays. Quantitative structure-toxicity relationships were established using exact mass and the final 60 selected molecular fingerprint features calculated from SMILES. To determine the optimal learning strategy, six regression algorithms were systematically compared, including partial least squares regression (PLS), elastic net regression, support vector regression (SVR), kernel ridge regression (KRR), random forest, and histogram-based gradient boosting. Model hyperparameter tuning was performed using repeated *k*-fold cross-validation, and the optimized models were evaluated on an independent test set using R², RMSE, and MAE. The best-performing model was selected as the final predictor.

The code for model training and testing, together with the optimized model, can be found in the folder:

`NHCs-MSTox_model_training`

---

## How Can Our Model Predict the Toxicity of Unknown MS/MS?

### Requirements

Before running the workflow, please make sure that:

- SIRIUS is installed
- Python and the required packages are installed

### Step 1. Calculate fingerprints using SIRIUS

1. Import the MS/MS file into the SIRIUS graphical user interface.  
   Example input file:  
   `Test example file_lminostilbene/Iminostilbene MS2.mgf`

2. Calculate the molecular fingerprints and export the fingerprint posterior probability file.  
   Example output file:  
   `Test example file_lminostilbene/Iminostilbene_fingerprint_posterior probability.json`

3. Reformat the posterior probability file strictly according to the input template file, and round all probabilities to 0 or 1.  
   Example template file:  
   `Test example file_lminostilbene/Iminostilbene_fingerprint.xlsx`

### Step 2. Predict toxicity values

Run the NHCs-MSTox graphical prediction program:

`NHCs-MSTox_prediction/NHCs-MSTox_predictor_gui.py`

Set the input file path and the output path for the prediction results, then click the **Run Prediction** button. The prediction results will be exported to the specified directory.

Example output file:  
`Test example file_lminostilbene/Iminostilbene_prediction_results.xlsx`

---

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.
