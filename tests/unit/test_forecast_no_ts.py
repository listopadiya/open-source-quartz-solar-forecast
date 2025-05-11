import pandas as pd
import pytest

from quartz_solar_forecast.forecast import predict_ocf, predict_tryolabs
from quartz_solar_forecast.weather.open_meteo import WeatherService
from quartz_solar_forecast.pydantic_models import PVSite


# Shared fixture for site input
@pytest.fixture
def site():
    return PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

# Shared fixture for time input
@pytest.fixture
def current_ts():
    return pd.Timestamp.now()

@pytest.fixture
def dummy_weatherservice(monkeypatch):
    start_time = datetime(2024, 6, 1, 0, 0)
    hours = pd.date_range(start=start_time, periods=3, freq='H')

    mock_weather_df = pd.DataFrame({
        "date": hours,
        "temperature_2m": [20.0, 21.5, 22.0],
        "relative_humidity_2m": [50, 52, 55],
        "dew_point_2m": [10, 11, 12],
        "precipitation": [0, 0, 0],
        "surface_pressure": [1010, 1011, 1012],
        "cloud_cover": [30, 25, 20],
        "cloud_cover_low": [10, 10, 5],
        "cloud_cover_mid": [10, 10, 5],
        "cloud_cover_high": [10, 5, 5],
        "wind_speed_10m": [3.0, 3.5, 4.0],
        "wind_direction_10m": [180, 190, 200],
        "is_day": [1, 1, 1],
        "shortwave_radiation": [500, 600, 700],
        "direct_radiation": [300, 400, 450],
        "diffuse_radiation": [200, 200, 250],
        "direct_normal_irradiance": [600, 650, 700],
        "terrestrial_radiation": [300, 310, 320],
    })

    # Monkeypatch the WeatherService method
    def mock_get_hourly_weather(self, latitude, longitude, start_date, end_date):
        return mock_weather_df

    monkeypatch.setattr(WeatherService, "get_hourly_weather", mock_get_hourly_weather)


# Run gradient boosting model with no ts
def test_predict_ocf_no_ts(site, current_ts):
    predictions_df = predict_ocf(site=site, model=None, ts=current_ts, nwp_source="icon")

    # check current ts agrees with dataset
    assert predictions_df.index.min() >= current_ts - pd.Timedelta(hours=1)

    print(predictions_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predictions_df['power_kw'].max()}")


# Run xgb model with no ts
def test_predict_tryolabs_no_ts(site, current_ts, dummy_weatherservice):    
    predictions_df = predict_tryolabs(site=site, ts=current_ts)

    # check current ts agrees with dataset
    assert predictions_df.index.min() >= current_ts - pd.Timedelta(hours=1)

    print(predictions_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predictions_df['power_kw'].max()}")
