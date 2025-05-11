import pandas as pd
import pytest
from datetime import datetime, timedelta

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
    # Monkeypatch the WeatherService method
    def mock_get_hourly_weather(self, latitude, longitude, start_date, end_date):
        print(f"MOCK IS CALLED {latitude} {longitude} {start_date} {end_date}")
        mock_hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=False),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=False),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(0).ValuesAsNumpy(),
            "dew_point_2m": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(0).ValuesAsNumpy(),
            "surface_pressure": hourly.Variables(0).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(0).ValuesAsNumpy(),
            "cloud_cover_low": hourly.Variables(0).ValuesAsNumpy(),
            "cloud_cover_mid": hourly.Variables(0).ValuesAsNumpy(),
            "cloud_cover_high": hourly.Variables(0).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(0).ValuesAsNumpy(),
            "is_day": hourly.Variables(0).ValuesAsNumpy(),
            "shortwave_radiation": hourly.Variables(0).ValuesAsNumpy(),
            "direct_radiation": hourly.Variables(0).ValuesAsNumpy(),
            "diffuse_radiation": hourly.Variables(0).ValuesAsNumpy(),
            "direct_normal_irradiance": hourly.Variables(0).ValuesAsNumpy(),
            "terrestrial_radiation": hourly.Variables(0).ValuesAsNumpy(),
        }

        mock_weather_df = pd.DataFrame(mock_hourly_data)
        mock_weather_df["date"] = pd.to_datetime(df["date"])
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
