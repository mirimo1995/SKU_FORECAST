import os
from typing import Optional

import pandas as pd
import numpy as np
import fire
import logging
from constants import LEAD_TIME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(data_path: str = "data.csv",
               output_path: Optional[str] = '', lead_time: Optional[int] = LEAD_TIME) -> None:
    # Read the data
    data = pd.read_csv(data_path)
    data["DATE"] = pd.to_datetime(data["DATE"])
    data["YEAR"] = pd.to_datetime(data["DATE"]).dt.year

    # Keep only data after 2017
    data = data[data.DATE.dt.year >= 2017]

    # Filter SKUs that have less than 30 appearances in the last 2 years
    yearly_appearances = data.groupby(['SKU', 'YEAR']).size().unstack(fill_value=0)
    last_two_years_appearances = yearly_appearances[[2023, 2024]].sum(axis=1)
    skus_to_keep = last_two_years_appearances[last_two_years_appearances >= 30].index
    data = data[data['SKU'].isin(skus_to_keep)]
    logger.info(f"Number of unique SKUs after filtering: {data.SKU.nunique()}")

    # Rearrange data to have all SKUs at each date
    date_range = pd.date_range(start=data['DATE'].min(), end=data['DATE'].max())
    all_skus = data['SKU'].unique()
    adjusted_data = pd.DataFrame(
        {'DATE': np.repeat(date_range, len(all_skus)), 'SKU': np.tile(all_skus, len(date_range))})
    adjusted_data = adjusted_data.merge(data[['DATE', 'SKU', 'QUANTITY_SOLD']], on=['DATE', 'SKU'], how='left')
    adjusted_data['QUANTITY_SOLD'].fillna(0, inplace=True)
    adjusted_data = adjusted_data.sort_values(by=['SKU', 'DATE']).reset_index(drop=True)
    adjusted_data['target'] = adjusted_data.groupby('SKU')['QUANTITY_SOLD'].transform(
        lambda x: x.shift(-lead_time).rolling(window=lead_time).sum())

    # Feature engineering
    def add_features(data, lead_time):
        data['target_last_year'] = data.groupby('SKU')['target'].shift(365)
        data['target_last_lead_time'] = data.groupby('SKU')['target'].shift(lead_time)

        data['lag_1_year'] = data.groupby('SKU')['QUANTITY_SOLD'].shift(365)
        data['lag_1_month'] = data.groupby('SKU')['QUANTITY_SOLD'].shift(30)
        data['lag_3_days'] = data.groupby('SKU')['QUANTITY_SOLD'].shift(3)
        data['lag_1_day'] = data.groupby('SKU')['QUANTITY_SOLD'].shift(1)
        # Moving averages to capture trend
        data['moving_avg_last_2_month'] = data.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.rolling(window=60).mean())
        data['moving_avg_last_month'] = data.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.rolling(window=30).mean())
        data['moving_avg_last_3_days'] = data.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.rolling(window=3).mean())
        data['moving_avg_last_5_days'] = data.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.rolling(window=5).mean())
        data['moving_avg_last_10_days'] = data.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.rolling(window=10).mean())
        data['moving_avg_last_year'] = data.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.rolling(window=365).mean())

        # Date features to capture seasonality and other temporal patterns
        data['month'] = data['DATE'].dt.month
        data['week'] = data['DATE'].dt.isocalendar().week

        # Global features based on all SKUs to capture overall trends
        data['global_sum_qty'] = data.groupby('DATE')['QUANTITY_SOLD'].transform('sum')
        data['global_moving_avg_last_month'] = data.groupby('DATE')['global_sum_qty'].transform(
            lambda x: x.rolling(window=30).mean())
        data['global_lag_last_month'] = data.groupby('DATE')['global_sum_qty'].shift(30)
        data['global_moving_avg_3_days'] = data.groupby('DATE')['global_sum_qty'].transform(
            lambda x: x.rolling(window=3).mean())
        data['global_lag_3_days'] = data.groupby('DATE')['global_sum_qty'].shift(3)
        data['global_lag_1_day'] = data.groupby('DATE')['global_sum_qty'].shift(1)
        data['global_moving_avg_last_week'] = data.groupby('DATE')['global_sum_qty'].transform(
            lambda x: x.rolling(window=7).mean())

        return data.dropna()

    data_ml = add_features(adjusted_data, lead_time)

    # Split data into train and test sets
    last_year_date = data_ml['DATE'].max() - pd.DateOffset(years=1)
    train_data = data_ml[data_ml['DATE'] <= last_year_date]
    test_data = data_ml[data_ml['DATE'] > last_year_date]

    # Save the train and test sets to CSV files
    train_data.sort_values(by=['SKU', 'DATE']).to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    test_data.sort_values(by=['SKU', 'DATE']).to_csv(os.path.join(output_path, "test_data.csv"), index=False)

    logger.info(f"Train and test data saved")


if __name__ == "__main__":
    fire.Fire(preprocess)