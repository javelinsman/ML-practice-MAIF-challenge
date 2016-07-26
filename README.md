# MAIF Challenge

My practice codes for [MAIF challenge](https://www.datascience.net/fr/challenge/26/details), which is open-competition of machine learning and data science. This practice was managed by Machine Learning course in 2016 HEIG-VD & SNU Summer University, Switzerland.

## Short Description

Infer the annual insuarance fee from other given variables(known or unknown).

* `data/ech_apprentisage.csv` Training Data
* `data/ech_test.csv` Test Data

## Shotgun & Ensemble Method

Using all given variables, train several models and merge the results using simple regressor. String variables are converted to binary strings.

`PATH_TO_ANACONDA/bin/python shotgun_and_ensemble.py first[second]`

The best result : 12.476 %

## Many Decision Trees Method

The experimental method which makes many decision trees, trained by randomly-selected samples from original training data, and merge their decision.

`PATH_TO_ANACONDA/bin/python many_decision_trees.py`

The best result : 16.557 %
