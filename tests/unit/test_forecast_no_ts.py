import pandas as pd
import pytest

from quartz_solar_forecast.forecast import predict_ocf, predict_tryolabs
from quartz_solar_forecast.pydantic_models import PVSite
from tests.unit.mocks import mock_weather_api


# Shared fixture for site input
@pytest.fixture
def site():
    return PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

# Shared fixture for time input
@pytest.fixture
def current_ts():
    return pd.Timestamp.now()


# Run gradient boosting model with no ts
def test_predict_ocf_no_ts(site, current_ts, mock_weather_api):
    predictions_df = predict_ocf(site=site, model=None, ts=current_ts, nwp_source="icon")

    # check current ts agrees with dataset
    assert predictions_df.index.min() >= current_ts - pd.Timedelta(hours=1)

    print(predictions_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predictions_df['power_kw'].max()}")


# Run xgb model with no ts
def test_predict_tryolabs_no_ts(site, current_ts, mock_weather_api):    
    predictions_df = predict_tryolabs(site=site, ts=current_ts)

    # check current ts agrees with dataset
    assert predictions_df.index.min() >= current_ts - pd.Timedelta(hours=1)

    print(predictions_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predictions_df['power_kw'].max()}")
