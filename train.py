import pandas as pd
import numpy as np
import xgboost as xgb
import fire

def train_model_sampled_smoothed(train_data_path, test_data_path, output_model_path):
    # Load train and test datasets
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    features = ['target_last_year',
       'target_last_lead_time', 'lag_1_year', 'lag_1_month', 'lag_3_days',
       'lag_1_day', 'moving_avg_last_2_month', 'moving_avg_last_month',
       'moving_avg_last_3_days', 'moving_avg_last_5_days',
       'moving_avg_last_10_days', 'moving_avg_last_year', 'month', 'week',
       'global_moving_avg_last_month',
       'global_lag_last_month', 'global_moving_avg_3_days',
       'global_lag_3_days', 'global_lag_1_day', 'global_moving_avg_last_week']
    train_data_sample = train_data.copy()

    zero_sale_data = train_data_sample[train_data_sample['moving_avg_last_month'] == 0]
    non_zero_sale_data = train_data_sample[train_data_sample['moving_avg_last_month'] != 0]

    sampled_zero_sale_data = zero_sale_data.sample(frac=0.2, random_state=42)

    balanced_train_data = pd.concat([non_zero_sale_data, sampled_zero_sale_data])

    # Hyperparameters
    best_params = {
        'n_estimators': 81,
        'max_depth': 6,
        'learning_rate': 0.03707824028156351,
        'subsample': 0.9,
        'colsample_bytree': 0.6,
        'gamma': 0.018309512586500896,
        'lambda': 0.00018860129880107608,
        'alpha': 0.002293722691755294}

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(balanced_train_data[features], balanced_train_data["target"])

    # Save the trained model
    xgb_model.save_model(output_model_path)

def main():
    train_model_sampled_smoothed("train_data.csv",
                                 "test_data.csv", output_model_path="output.model")

if __name__ == "__main__":
    fire.Fire(train_model_sampled_smoothed)