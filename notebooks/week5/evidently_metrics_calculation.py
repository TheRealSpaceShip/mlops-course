"""
Evidently metrics calculation
"""

import datetime
import calendar
import time
import random
import logging
import pickle
import pandas as pd
import psycopg

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnQuantileMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

CREATE_TABLE_STATEMENT = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	fare_amount_quantile float
)
"""

YEAR = 2024
MONTH = 3
begin_date = datetime.datetime(year=YEAR, month=MONTH, day=1)
_, last_day = calendar.monthrange(year=YEAR, month=MONTH)


def load_bin(filename):
    """Load binary file dump with pickle"""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


model = load_bin("models/lin_reg.bin")
dv = load_bin("models/dv.bin")

raw_data = pd.read_parquet(f"data/green_tripdata_{YEAR:04d}-{MONTH:02d}.parquet")
reference_data = pd.read_parquet("data/reference.parquet")

numerical_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
categorical_features = ["PULocationID", "DOLocationID"]
column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target=None,
)

raw_data["duration"] = raw_data.lpep_dropoff_datetime - raw_data.lpep_pickup_datetime
raw_data.duration = raw_data.duration.apply(lambda td : float(td.total_seconds())/60)
raw_data = raw_data[(raw_data.duration >= 0) & (raw_data.duration <= 60)]
raw_data = raw_data[(raw_data.passenger_count > 0) & (raw_data.passenger_count <= 8)]
raw_data[categorical_features] = raw_data[categorical_features].astype(str)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task
def prep_db():
    """Prepare database"""
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(CREATE_TABLE_STATEMENT)


@task
def calculate_metrics_postgresql(curr, start_date, end_date):
    """Calculate and store metrics to database"""
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= start_date)
        & (raw_data.lpep_pickup_datetime < end_date)
    ]

    if current_data.empty:
        return

    dicts = current_data[categorical_features + numerical_features].to_dict(
        orient="records"
    )

    data = dv.transform(dicts)

    # current_data.fillna(0, inplace=True)
    current_data["prediction"] = model.predict(data)

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    fare_amount_quantile = result["metrics"][1]["result"]["current"]["value"]
    num_drifted_columns = result["metrics"][2]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][3]["result"]["current"][
        "share_of_missing_values"
    ]

    curr.execute(
        "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, fare_amount_quantile) values (%s, %s, %s, %s, %s)",
        (
            start_date,
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            fare_amount_quantile,
        ),
    )


@flow
def batch_monitoring_backfill():
    """Batch monitoring"""
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for i in range(0, last_day):
            start_date = begin_date + datetime.timedelta(i)
            end_date = begin_date + datetime.timedelta(i + 1)
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, start_date, end_date)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
