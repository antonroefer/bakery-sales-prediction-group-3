# Dataset Characteristics

The dataset exploration can be seen in Data_merging_and_initial_exploration.ipynb.

We played around with temperature dependend features in temperature_konstantin.ipynb. We tried to group our temperature, but in the end sticked to temperature as itself as a feature. We even tried to calculate a temperature difference feature: temperature - seasonal temperature in 0_Data_Preparation/old_interpolation_wetter.py, where we used a savgol filter of 200 days to calculate a seasonal temperature.

We dealt with missing values in cloud coverage, temperature and wind velocity with KNN imputation and one hot encoding the 6 most prominent weather_codes with more than 600 entries in the entire dataset. This can be seen in missing_values.ipynb.