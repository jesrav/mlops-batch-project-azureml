"""
Module to get raw dataset.
"""
from pathlib import Path
from typing import Optional
import logging

import hydra
from hydra.utils import get_original_cwd
from sklearn.datasets import fetch_california_housing
import pandas as pd

logger = logging.getLogger(__name__)


def get_raw_data(
    sample_size: Optional[int] = None,
    med_inc_mean_drift_percentage: Optional[float] = None, 
    **kwargs
    ) -> pd.DataFrame:
    """Get california housing data."""
    _ = kwargs
    data = fetch_california_housing(as_frame=True)
    df = data.data
    df["median_house_price"] = data.target
    if med_inc_mean_drift_percentage:
        df["MedInc"] = df["MedInc"] * (1 + med_inc_mean_drift_percentage)
    if sample_size:
        return df.sample(sample_size)
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    logger.info("Get raw training data")
    df = get_raw_data(
        sample_size=config["main"].get("inference_sample_size", None),
        med_inc_mean_drift_percentage=config["main"].get("med_inc_mean_drift_percentage", None)
    )

    out_path = Path(get_original_cwd()) / config['data']['raw_data']
    logger.info(f"Save raw data to {out_path}")
    df.to_parquet(out_path)


if __name__ == "__main__":
    main()





