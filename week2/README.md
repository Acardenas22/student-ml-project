\# Week 2 — ML Model Training + Evaluation (Proof of Concept)



\## Goal

Train, evaluate, and serialize a machine learning model that predicts a \*\*quality/compliance-like score\*\* from structured numeric inputs.  

This week serves as a \*\*proof of concept\*\* for integrating an ML “scoring + remediation suggestion” component into our NEC compliance validator pipeline.



\## Dataset

\*\*Wine Quality (Red)\*\* dataset (UCI).  

\- Rows: 1,599  

\- Features: 11 numeric inputs (e.g., acidity, sulphates, alcohol)  

\- Target: `quality` (integer score)



\### Why this dataset?

We used this dataset as a \*\*proxy\*\* because it is:

\- clean and well-known (low data friction for a PoC)

\- fully numeric (FastAPI-friendly inputs)

\- suitable for regression and regularization experiments (Linear/Ridge/Lasso)



In the NEC context, the features represent structured design parameters, and the target represents an overall \*\*compliance/quality score\*\*.



\## Approach

1\. Load dataset (`winequality-red.csv`)

2\. Split into train/test sets

3\. Scale features using `StandardScaler`

4\. Train and compare:

&nbsp;  - Linear Regression

&nbsp;  - Ridge Regression

&nbsp;  - Lasso Regression

5\. Select best model using \*\*test RMSE\*\* (lower is better)

6\. Save model and scaler for deployment



\## Results

Test set metrics:



\- Linear Regression → RMSE: \*\*0.6245\*\*, R²: \*\*0.4032\*\*

\- Ridge Regression → RMSE: \*\*0.6245\*\*, R²: \*\*0.4032\*\*

\- Lasso Regression → RMSE: \*\*0.6627\*\*, R²: \*\*0.3279\*\*



\*\*Selected model:\*\* Linear Regression  

Linear and Ridge performed identically, suggesting regularization did not provide additional benefit for this dataset at the chosen parameters.



\## Artifacts Saved

After training, artifacts are saved to:



model/

├─ model.joblib

└─ scaler.joblib


These artifacts are intended to be loaded by the FastAPI service in Week 3.



\## How to Run



\### Install dependencies

```bash

pip install pandas scikit-learn joblib

TRAIN THE MODEL

python train\_model.py


SMOKE TEST PREDICTIONS

python predict.py





LIMITATIONS AND FUTURE WORK



The Wine Quality dataset is used as a proxy to validate the ML pipeline.



Real NEC compliance data may be noisier and less structured.



Future work includes replacing the proxy dataset with NEC-labeled or synthetic NEC compliance data.

