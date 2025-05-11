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
    # Monkeypatch get_hourly_weather method
    def mock_get_hourly_weather(self, latitude, longitude, start_date, end_date):
        print(f"MOCK IS CALLED {latitude} {longitude} {start_date} {end_date}")
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "is_day",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation",
        ]
        mock_hourly_date = pd.date_range(
        	start = pd.to_datetime(start_date, format="%Y-%m-%d", utc = True),
        	end = pd.to_datetime(end_date, format="%Y-%m-%d", utc = True) + pd.Timedelta(days=1),
        	freq = pd.Timedelta(hours=1),
        	inclusive = "left"
        )
        mock_weather_df = pd.DataFrame({ "date": mock_hourly_date })
        for v in variables:
            mock_weather_df[v] = 0.0
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
    print(predictions_df)
    # check current ts agrees with dataset
    assert predictions_df.index.min() >= current_ts - pd.Timedelta(hours=1)

    print(predictions_df)
    print(f"Current time: {current_ts}")
    print(f"Max: {predictions_df['power_kw'].max()}")
