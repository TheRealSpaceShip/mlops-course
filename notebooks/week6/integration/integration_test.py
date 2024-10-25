import pandas as pd
import batch as bt
import os
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_integration():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    input_file = bt.get_input_path(2023, 1)
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
    storage_options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}
    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=storage_options
    )
