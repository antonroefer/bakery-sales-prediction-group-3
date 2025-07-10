# Baseline Model

For our Baseline Model we used basemodel_GA.py for our kaggle submissions firstly, but after data imputation, we created a new model basemodel_roefer.py, which improved the kaggle score a bit.

All submission CSVs from this model, which can be found in kaggle, are in the folder "submissions" aswell.

The feature selection was not based on the results of the base model, but on the results of the neural net. Actually including non significant features in the linear model, lead to better MAPE scores in some runs of older versions of our neural net.