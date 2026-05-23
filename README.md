# NHCs-MSTox

<p align="center">
  <img src="docs/Fig1.jpg" alt="Fig1" width="1200"/>
</p>

## Training of NHCs-MSTox model

NHCs-MSTox is a machine-learning framework developed for the toxicity prediction of nitrogen-containing heterocyclic compounds (NHCs). The model was trained on an in-house *Vibrio fischeri* acute toxicity dataset and uses exact mass together with molecular fingerprint features to establish quantitative structure-toxicity relationships. To identify the optimal learning strategy, six regression algorithms were systematically compared, including PLS, elastic net, SVR, KRR, random forest, and histogram-based gradient boosting. After repeated cross-validation and independent test-set evaluation, the best-performing model was selected as the final predictor for downstream application to unknown compounds and transformation products.

## How can our models predict the toxicity of unknown MS/MS?

Our workflow enables toxicity prediction even when a compound has not been fully identified. First, LC-HRMS/MS data are converted into model-compatible structural features. During model development, standardized SMILES were used to calculate molecular fingerprints and exact mass. During practical application, unknown MS/MS spectra are processed to infer the same type of fingerprint information, which is then aligned to the final model input space. The resulting feature vector, composed of 60 selected molecular fingerprints plus exact mass, is fed into the trained NHCs-MSTox model to predict pEC50 values. In this way, the framework supports rapid toxicity screening, prioritization, and ranking of unknown NHC-related compounds directly from MS/MS-derived information.
