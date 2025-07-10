# Dataset Characteristics

The dataset exploration can be seen in the [Characteristics Notebook](characteristics.ipynb).

We played around with temperature dependend features in [temperature.ipynb](temperature.ipynb). We tried to group our temperature, but in the end *sticked to temperature* as itself as a feature.
We even tried to calculate a temperature difference feature: temperature - seasonal temperature in [0_Data_Preparation/archived/old_interpolation_wetter.py](../0_Data_Preparation/archived/old_interpolation_wetter.py), where we used a savgol filter of 200 days to calculate a seasonal temperature. But all these temperature difference features were not significant in our base model.

We dealt with missing values in cloud coverage, temperature and wind velocity with KNN imputation and one hot encoding the 6 most prominent weather_codes with more than 500 entries in the entire dataset. This can be seen in the [Missing Values Notebook](missing_values.ipynb). We also tried out IterativImputer, but the kaggle score got worse when doing that, so we sticked to KNN.

The datasets without missing values are in the folder: [imputed_data](imputed_data), where we used [data_after_imputation_knn.csv](imputed_data/data_after_imputation_knn.csv)*