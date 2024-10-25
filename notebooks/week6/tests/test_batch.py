import batch as bt
import pandas as pd
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # Columns to be treated as categorical
    categorical = ["PULocationID", "DOLocationID"]

    # Call function
    processed_df = bt.prepare_data(df, categorical)

    # Assertions
    assert len(processed_df) == 2  # Only two durations are valid (and within the [1, 60] min range)
    assert (processed_df['duration'] >= 1).all() and (processed_df['duration'] <= 60).all()
    assert df.dtypes['PULocationID'] == 'float64'
    assert df.dtypes['DOLocationID'] == 'float64'
    # Check that missing values in categorical columns have been replaced with '-1'
    assert (processed_df['PULocationID'] == '-1').any()
    assert (processed_df['DOLocationID'] == '-1').any()
