# Baseline Model

For our Baseline Model we used [basemodel_GA.py](./basemodel_GA.py) for our kaggle submissions firstly, but after data imputation, we created a new model [basemodel_roefer.py](./basemodel_roefer.py), which improved the kaggle score a bit.

All submission CSVs from this model, which can be found in kaggle, are in the folder ["submissions" ](./submissions/)aswell.

The feature selection was not based on the results of the base model, but on the results of the neural net. That is why we included all our variables in the basemodel. Actually including linearly non significant features, lead to better MAPE scores in some runs of older versions of our neural net, when we still used a sigmoid activation function.

The ["random forest regression.py"](random%20forest%20regression.py) script helped us to select important features in the neural net, which is why we included it in the [BaselineModel Folder](../2_BaselineModel/).