We built a Neural Network using Keras to predict bakery sales (Umsatz). The Jupyter Noterbook includes the steps for data preparation, building and training the model, and evaluating its performance — overall and by product group (Warengruppe).

# Data preparation

We manually removed columns that we considered less relevant or potentially noisy, such as:

-Calendar variables: Silvester, Feiertage, Wahltag, Advent, etc.
-Weather variables: Bewoelkung, Niederschlag, some Wettercode_* columns

This step helped simplify the model and focus on more useful predictors. Based on experimentation our best performing attempt was the one where
we kept the  Temperatur, Schulferien, Weekday, days_to_silvester and Warengruppe_1 to Warengruppe_6 variables.

Before training, all input features were scaled using StandardScaler to standardize their ranges.


# Model characteristics

* An input layer matched to the number of features
* Three hidden layers with 128, 64, and 32 neurons, using ReLU activation. L2 regularization was applied to the first hidden layer to reduce overfitting
* An output layer with one neuron
* Loss function: Mean Squared Error (MSE)
* Optimizer: Adam
* Learning Rate:0.0001
* Batch Size:32
* Epochs:600
* Early stopping to stop training when validation loss stopped improving, with a patience of 50 epochs

# Evaluation

We calculated the Mean Absolute Percentage Error (MAPE) to evaluate the prediction accuracy on the full training and validation dataset, and separately for each product group (Warengruppe). Additionally, by visualizing the MAPE values across training and validation sets and among the different product groups, we were able to identify which product groups the model predicts better and which require further improvement.


