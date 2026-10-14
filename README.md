# DSCI 552 — Homework 4

**Om Bhatt** · USC, DSCI 552 (Machine Learning for Data Science)

Time-series classification of human activities from the
[AReM dataset](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem)
(UCI), plus ISLR exercises 4.8.3 and 4.8.7.

## Files

| File | Description |
|---|---|
| [`Bhatt_Om_HW4.ipynb`](Bhatt_Om_HW4.ipynb) | Main notebook: code, outputs, plots and written answers |
| [`Bhatt_Om_HW4.html`](Bhatt_Om_HW4.html) | Same notebook exported to HTML (download and open in a browser) |
| `data/AReM/` | AReM data: 7 activity folders, 88 instances × 6 time series × 480 samples |
| `requirements.txt` | Python packages used |

## What's in the notebook

1. **Feature extraction (from HW3):** load and clean the data (a few files have a different delimiter or trailing commas), split it into train/test (69/19), and extract time-domain features (mean, std, max) after breaking each series into `l` pieces.
2. **(a) Binary classification (bending vs. other) with logistic regression:**
   scatter plots for `l = 1, 2`; 5-fold **stratified** CV over `(l, p)` with recursive feature elimination redone *inside* every fold (the right way to cross-validate); confusion matrix, ROC/AUC, β's and Wald p-values; test-set evaluation; a discussion of separation and instability; **case-control sampling** with the intercept correction.
3. **(b) L1-penalized logistic regression** (liblinear), cross-validating `l` and `C`, compared with the RFE/p-value approach.
4. **(c) Multi-class classification:** L1-penalized multinomial regression (saga), Gaussian and Multinomial Naïve Bayes; multi-class confusion matrices and one-vs-rest ROC curves (micro/macro AUC).
5. **ISLR 4.8.3:** proof that the Bayes classifier is quadratic when the class variances differ.
6. **ISLR 4.8.7:** P(dividend | X = 4) ≈ **0.752**.

### Main results

| Task | Model | CV accuracy | Test accuracy |
|---|---|---|---|
| Binary | Logistic + RFE (l = 3, p = 1) | 0.986 | 0.842 |
| Binary | L1 logistic (l = 12, 5 features) | 0.986 | **0.947** |
| Multi-class | L1 multinomial (l = 17) | 0.900 | 0.737 |
| Multi-class | Gaussian NB (l = 1) | 0.900 | **0.895** |
| Multi-class | Multinomial NB (l = 10) | 0.799 | 0.789 |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Bhatt_Om_HW4.ipynb
```

Run all cells from the repository root (the notebook reads `data/AReM/`). A full run takes about 5 minutes, mostly the cross-validation in 2(a)iii and 2(c). Results are reproducible (`SEED = 42`).
