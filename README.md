# DSCI 552 HW4 - Activity Recognition (AReM)

**Name:** Om Bhatt<br>
**USC ID:** 7536194801

Time series classification part 2: binary and multi-class classification on the UCI AReM dataset, plus ISLR 4.8.3 and 4.8.7.

A person wore wireless sensors on the chest and both ankles while doing 7 activities
(bending1, bending2, cycling, lying, sitting, standing, walking).
Each `datasetN.csv` file is one instance: 6 time series of 480 values
(avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23, var_rss23). There are 88 instances (69 train / 19 test).

All the work is in [notebooks/bhatt_om_hw4.ipynb](notebooks/bhatt_om_hw4.ipynb), which is the notebook to grade. It's already run, so the outputs and plots show up on GitHub.
[notebooks/bhatt_om_hw4.html](notebooks/bhatt_om_hw4.html) is the same notebook exported to HTML.

## Contents

- `notebooks/bhatt_om_hw4.ipynb` - main assignment notebook (+ HTML export)
- `data/AReM/` - dataset, the 7 activity folders
- `output/` - CSV outputs from the notebook:
  - `binary_rfe_cv.csv` - CV accuracy for every (l, p) with logistic regression + RFE
  - `binary_l1_cv.csv` - best C and CV accuracy for every l, L1 logistic regression
  - `multiclass_l1_cv.csv` - best C and CV accuracy for every l, L1 multinomial regression
  - `multiclass_nb_cv.csv` - best smoothing parameter and CV accuracy for every l, Gaussian and Multinomial NB
  - `multiclass_summary.csv` - test error and AUCs of the three multi-class models

## What's in the notebook

Problem 1: Time Series Classification Part 1 (HW3 code, re-used because part 2 needs it)

- load the 88 instances (the loader handles the messy files), train/test split, and mean / std / max of each series after breaking it into `l` pieces

Problem 2: Time Series Classification Part 2

- (a) i-ii. scatter plots of mean, std, max of series 1, 2, 6 for l = 1 and l = 2
- (a) iii. 5-fold stratified CV over (l, p), with scaling and recursive feature elimination redone inside every fold (the right way to cross-validate)
- (a) iv. final model: confusion matrix, ROC/AUC, coefficients and Wald p-values
- (a) v. test set: accuracy 0.842 vs. CV accuracy 0.986
- (a) vi. separation: the full feature sets are linearly separable, so the logistic regression coefficients are unstable
- (a) vii. class imbalance (9 vs 60) and case-control sampling with the intercept correction
- (b) L1-penalized logistic regression (liblinear), CV over l and C, compared with RFE
- (c) i. L1-penalized multinomial regression (saga), multi-class confusion matrix and one-vs-rest ROC curves
- (c) ii-iii. Gaussian vs. Multinomial Naive Bayes, and which method is best

Problem 3: ISLR 4.8.3 - with class-specific variances the Bayes classifier is quadratic, not linear

Problem 4: ISLR 4.8.7 - P(dividend | X = 4) = 0.752

## Results

| Task | Model | CV accuracy | Test accuracy |
|---|---|---|---|
| Binary | Logistic + RFE (l = 3, p = 1) | 0.986 | 0.842 |
| Binary | L1 logistic (l = 12, 5 features) | 0.986 | 0.947 |
| Multi-class | L1 multinomial (l = 17) | 0.900 | 0.737 |
| Multi-class | Gaussian NB (l = 1) | 0.900 | 0.895 |
| Multi-class | Multinomial NB (l = 10) | 0.799 | 0.789 |

Some things I found:
- One feature, `mean5_1` (mean of avg_rss23 over the first third of the recording), is enough for 98.6% CV accuracy on bending vs. other.
- The test accuracy is lower than the CV accuracy because the test set is tiny (19 instances, one error = 5.3%) and (l, p) was picked as the best of ~590 CV scores.
- L1 logistic regression did better on the test set than RFE and was easier to set up.
- For multi-class, the simplest model (Gaussian NB) won. Almost all errors are sitting vs. standing.

## Running it

```
pip install -r requirements.txt
jupyter notebook notebooks/bhatt_om_hw4.ipynb
```

The notebook reads the data from `data/AReM/` and saves the CSV outputs to `output/` (paths are relative to the `notebooks/` folder). A full run takes about 5 minutes. Results are reproducible (`SEED = 42`).

## Data

UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem

- Palumbo, F., Gallicchio, C., Pucci, R., & Micheli, A. (2016). Activity Recognition system based on Multisensor data fusion (AReM) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS33
- Filippo Palumbo, Claudio Gallicchio, Rita Pucci, Alessio Micheli, Human activity recognition using multisensor data fusion based on Reservoir Computing,
  Journal of Ambient Intelligence and Smart Environments, Volume 8, Number 2, 2016, Pages 87-107.

## AI usage disclaimer

I used Claude only to understand the concepts and the assignment, to set up a format, to ask general programming questions, and to generate sample code that explains how certain programming constructs work.
Claude Code was used to add explanatory comments to some notebook cells, which also helped me understand the concepts. The analysis and code are my own.
