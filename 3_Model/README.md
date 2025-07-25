We built a Neural Network using Keras to predict bakery sales (Umsatz). The [Jupyter Noterbook](./Neural_Network_Group3.ipynb) includes the steps for additional data preparation, building and training the model, and evaluating its performance — overall and by product group (Warengruppe).

# Additional data preparation

We manually removed columns step by step that either we considered less relevant and potentially noisy or that were suggested irrelevant by our [random forest regression](./../2_BaselineModel/random%20forest%20regression.py), such as some calendar variables (Silvester, Feiertage, Wahltag, Advent) and weather variables (Bewoelkung, Niederschlag).

These steps helped to simplify the model and focus on more useful predictors. Based on experimentation our best performing attempt was the one where
we kept the Temperatur, Schulferien, Weekday, days_to_silvester and Warengruppe_1 to Warengruppe_6 variables.

Before training, all input features were scaled using StandardScaler to standardize their ranges.

# Hyper parameter tuning

Our Hyperparameter tuning was a quite similar process as the feature selection: We started with a quite complex model with up to 128 Neurons per Hidden Layer, with a sigmoid activation function for the first Layer and ReLU for the others, aswell as L2 regularization for each Hidden Layer and 3 DropOut Layers in between the Hidden Layers - suggested by ChatGPT.

Step by step, we dropped the DropOut Layers (pun intended), aswell as the HiddenLayer with the sigmoid activation which lead to more reliable results and smoother loss curves. With sigmoid we had these little plateaus for the first few epochs, which can be seen in this [figure](./archived/loss%20plots/Keras_N_OB_3.png) for example.

Our final and best model can be seen below:

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


