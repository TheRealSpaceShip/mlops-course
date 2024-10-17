#!/usr/bin/env python
# coding: utf-8

"""Script for week4 of MLOps course"""

import pickle
import sys
import pandas as pd


year = int(sys.argv[1]) # 2023
month = int(sys.argv[2]) # 4

MODEL_FILE = 'model.bin'
# pylint: disable=line-too-long
DATA_FILE_URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
OUTPUT_FILE = 'output.parquet'

CATEGORICAL = ['PULocationID', 'DOLocationID']


def read_data(filename):
    """Read parquet file and return pandas DataFrame"""
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

data_frame = read_data(DATA_FILE_URL)
dicts = data_frame[CATEGORICAL].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f'The mean predicted duration: {y_pred.mean():0.3f}')

# df_result = pd.DataFrame()
# df_result['ride_id'] = f'{year:04d}/{month:02d}_' + data_frame.index.astype('str')
# df_result['prediction'] = y_pred

# df_result.to_parquet(
#     OUTPUT_FILE,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
