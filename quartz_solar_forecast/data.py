""" Function to get NWP data and create fake PV dataset"""
import ssl
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from quartz_solar_forecast.pydantic_models import PVSite
from quartz_solar_forecast.weather import WeatherService


def get_nwp(site: PVSite, ts: datetime, nwp_source: str = "icon") -> xr.Dataset:
    """
    Get GFS NWP data for a point time space and time

    :param site: the PV site
    :param ts: the timestamp for when you want the forecast for
    :param nwp_source: the nwp data source. Either "gfs", "icon" or "ukmo". Defaults to "icon"
    :return: nwp forecast in xarray
    """
    now = datetime.now()

    # Define the variables we want (and aliases). Visibility is handled separately after the main request
    variable_map = {
        "temperature_2m": "t",
        "precipitation": "prate",
        "cloud_cover_low": "lcc",
        "cloud_cover_mid": "mcc",
        "cloud_cover_high": "hcc",
        "wind_speed_10m": "si10",
        "shortwave_radiation": "dswrf",
        "direct_radiation": "dlwrf"
    }

    start = ts.date()
    end = start + pd.Timedelta(days=7)

    # Check whether the time stamp is more than 3 months in the past
    # If yes, load data from open-meteo Historical Weather API
    if (now - ts).days > 90:
        print("Warning: The requested timestamp is more than 3 months in the past. The weather data are provided by a reanalyse model and not ICON or GFS.")
        api_type = "archive"

    else:
        # Getting NWP from open meteo weather forecast API by ICON, GFS, or UKMO within the last 3 months
        api_type = {
            "icon": "dwd-icon",
            "gfs": "gfs",
            "ukmo": "forecast"
        }.get(nwp_source)
        if not api_type:
            raise Exception(f'Source ({nwp_source}) must be either "icon", "gfs", or "ukmo"')

    # Make call to open-meteo via wrapper
    weather_service = WeatherService()
    weather_data = weather_service.get_hourly_weather(
        latitude=site.latitude,
        longitude=site.longitude,
        start_date=f"{start}",
        end_date=f"{end}",
        variables=list(variable_map.keys()),
        api_type=api_type,
        model="ukmo_seamless" if nwp_source == "ukmo" else None
    )

    # Handle visibility
    if (now - ts).days <= 90:
        # Load data from open-meteo gfs model
        visibility_data = weather_service.get_hourly_weather(
            latitude=site.latitude,
            longitude=site.longitude,
            start_date=f"{start}",
            end_date=f"{end}",
            variables=["visibility"],
            api_type="gfs"
        )
        weather_data["vis"] = visibility_data["visibility"].values
    else:
        # Set to maximum visibility possible
        weather_data["vis"] = 24000.0

    # Rename variable columns for future needs
    weather_data.rename(columns=variable_map, inplace=True)
    weather_data.rename(columns={"date": "time"}, inplace=True)
    weather_data = weather_data.set_index("time").astype('float64')
    print(weather_data)
    # Convert data into xarray
    data_xr = format_nwp_data(weather_data, nwp_source, site)

    return data_xr

def format_nwp_data(df: pd.DataFrame, nwp_source:str, site: PVSite):
    data_xr = xr.DataArray(
        data=df.values,
        dims=["step", "variable"],
        coords=dict(
            step=("step", df.index - df.index[0]),
            variable=df.columns,
        ),
    )
    data_xr = data_xr.to_dataset(name=nwp_source)
    data_xr = data_xr.assign_coords(
        {"x": [site.longitude], "y": [site.latitude], "time": [df.index[0]]}
    )
    return data_xr


def process_pv_data(live_generation_kw: Optional[pd.DataFrame], ts: pd.Timestamp, site: 'PVSite') -> xr.Dataset:
    """
    Process PV data and create an xarray Dataset.

    :param live_generation_kw: DataFrame containing live generation data, or None
    :param ts: Current timestamp
    :param site: PV site information
    :return: xarray Dataset containing processed PV data
    """
    if live_generation_kw is not None and not live_generation_kw.empty:
        # Get the most recent data
        recent_pv_data = live_generation_kw[live_generation_kw['timestamp'] <= ts]
        power_kw = np.array([recent_pv_data["power_kw"].values], dtype=np.float64)
        timestamp = recent_pv_data['timestamp'].values
    else:
        # Make fake PV data; this is where we could add the history of a PV system
        power_kw = np.array([[np.nan]])
        timestamp = [ts]

    da = xr.DataArray(
        data=power_kw,
        dims=["pv_id", "timestamp"],
        coords=dict(
            longitude=(["pv_id"], [site.longitude]),
            latitude=(["pv_id"], [site.latitude]),
            timestamp=timestamp,
            pv_id=[1],
            kwp=(["pv_id"], [site.capacity_kwp]),
            tilt=(["pv_id"], [site.tilt]),
            orientation=(["pv_id"], [site.orientation]),
        ),
    )
    da = da.to_dataset(name="generation_kw")

    return da

def make_pv_data(site: PVSite, ts: pd.Timestamp) -> xr.Dataset:
    """
    Make PV data by combining live data from various inverters.
    
    :param site: the PV site
    :param ts: the timestamp of the site
    :return: The combined PV dataset in xarray form
    """
    live_generation_kw = site.get_inverter().get_data(ts)
    # Process the PV data
    da = process_pv_data(live_generation_kw, ts, site)

    return da
