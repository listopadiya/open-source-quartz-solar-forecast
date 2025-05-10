import datetime
import pandas as pd
import zipfile
import os.path
import shutil
import logging

from huggingface_hub import hf_hub_download
from quartz_solar_forecast.weather import WeatherService

from xgboost.sklearn import XGBRegressor

from . import v2_config
import quartz_solar_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TryolabsSolarPowerPredictor:
    """
    A class to predict solar power output based on weather data, location, panel orientation,
        and other factors.

    Attributes
    ----------
    model_path : str
        Path to the trained model joblib file.

    Methods
    -------
    load_model -> None:

        Downloads the model from Google Drive and decropresses it if necessary
    
    predict_power_output(latitude: float, longitude: float, start_date: str, kwp: float,
        orientation: float, tilt: float) -> pd.DataFrame:

        Predicts solar power output for the given parameters.
    """
    DATE_COLUMN = "date"
    download_dir = os.path.dirname(quartz_solar_forecast.__file__)
    
    def _decompress_zipfile(self, filename: str) -> None:
        """
        Extracts the contents of a zip file to a given directory.

        Parameters:
        ----------
        filename : str
            The name of the .zip file to be decompressed.
        """
        # Get the directory of the file
        directory = os.path.dirname(filename)

        logger.info(f"Decompressing {filename} to {directory}")
        with zipfile.ZipFile(filename, "r") as zip_file:
            zip_file.extractall(path=directory)

    def load_model(
        self, 
        model_name: str = v2_config.MODEL_NAME,
        repo_id: str = v2_config.REPO_ID,
        file_path: str = v2_config.FILE_PATH
    ) -> XGBRegressor:
        """
        Downloads, prepares, and loads the XGBoost model for solar power prediction.
    
        Parameters:
        -----------
        model_name : str, optional
            The name of the model file (without .zip extension).
            Default is set in v2_config.MODEL_NAME.
        repo_id : str, optional
            The ID of the Hugging Face repository containing the model.
            Default is set in v2_config.REPO_ID.
        file_path : str, optional
            The path to the model file within the repository.
            Default is set in v2_config.FILE_PATH.
    
        Returns:
        --------
        XGBRegressor
            The loaded XGBoost model ready for making predictions.
        """        
        # Use the project directory instead of the user's home directory
        os.makedirs(self.download_dir, exist_ok=True)

        # No need to check if model is downloaded already,
        # as hf_hub_download does its own caching
        logger.info("Downloading model...")
        zip_path = hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=self.download_dir)

        # Decompress model file if it wasn't done yet
        model_path = os.path.join(self.download_dir, "models/v2", model_name)
        if not os.path.isfile(model_path):
            self._decompress_zipfile(zip_path)

        logger.info(f"Loading model from {model_path}")
        loaded_model = XGBRegressor()
        loaded_model.load_model(model_path)
        self.model = loaded_model
        return loaded_model
        
    def get_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        kwp: float,
        orientation: float = 180,
        tilt: float = 30,
    ) -> pd.DataFrame:
        """
        Fetches weather data for the given location and date range, and prepares it for prediction.

        Parameters
        ----------
        latitude : float
            Latitude of the location.
        longitude : float
            Longitude of the location.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        kwp : float
            Kilowatt peak of the solar panel system.
        orientation : float
            Orientation angle of the solar panel system in degrees.
        tilt : float
            Tilt angle of the solar panel system in degrees.

        Returns
        -------
        pd.DataFrame
            Prepared weather data with additional solar panel parameters.
        """
        start_date_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_datetime = start_date_datetime + datetime.timedelta(days=2)
        end_date = end_date_datetime.strftime("%Y-%m-%d")

        weather_service = WeatherService()

        weather_data = weather_service.get_hourly_weather(
            latitude, longitude, start_date, end_date
        )

        PANEL_COLUMNS = [
            "latitude_rounded",
            "longitude_rounded",
            "orientation",
            "tilt",
            "kwp",
        ]

        weather_data["latitude_rounded"] = latitude
        weather_data["longitude_rounded"] = longitude
        weather_data["orientation"] = orientation
        weather_data["tilt"] = tilt
        weather_data["kwp"] = kwp

        cols = PANEL_COLUMNS + [
            col for col in weather_data.columns if col not in PANEL_COLUMNS
        ]
        weather_data = weather_data[cols]

        return weather_data

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and transforms the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the weather and solar panel data.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame ready for prediction.
        """
        df.loc[:, self.DATE_COLUMN] = pd.to_datetime(df[self.DATE_COLUMN])

        df.loc[:, "date_year"] = df[self.DATE_COLUMN].dt.year
        df.loc[:, "date_month"] = df[self.DATE_COLUMN].dt.month
        df.loc[:, "date_day"] = df[self.DATE_COLUMN].dt.day
        df.loc[:, "date_hour"] = df[self.DATE_COLUMN].dt.hour
        df.loc[:, "date_minute"] = df[self.DATE_COLUMN].dt.minute

        COLUMNS_TO_DROP = ["date_minute", "date_year",
                           "terrestrial_radiation",
                           "shortwave_radiation",
                           "direct_normal_irradiance"]

        df = df.drop(columns=COLUMNS_TO_DROP)

        return df

    def predict_power_output(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        kwp: float,
        orientation: float = 180,
        tilt: float = 30,
    ) -> pd.DataFrame:
        """
        Predicts solar power output for the specified parameters.

        Parameters
        ----------
        latitude : float
            Latitude of the location.
        longitude : float
            Longitude of the location.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        kwp : float
            Kilowatt peak of the solar panel system.
        orientation : float
            Orientation angle of the solar panel system in degrees.
        tilt : float
            Tilt angle of the solar panel system in degrees.

        Returns
        -------
        pd.DataFrame
            DataFrame containing timestamps and predicted power output in kW for every 15 minutes.
        """

        data = self.get_data(latitude, longitude, start_date, kwp, orientation, tilt)
        #if data is not None:
        cleaned_data = self.clean(data)
        predictions = self.model.predict(cleaned_data.drop(columns=[self.DATE_COLUMN]))
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        final_data = cleaned_data.join(predictions_df)
        # set night predictions to 0
        final_data.loc[final_data["is_day"] == 0, "prediction"] = 0
        # set negative output to 0
        final_data.loc[final_data["prediction"] < 0, "prediction"] = 0
        df = final_data[[self.DATE_COLUMN, "prediction"]]
        df = df.rename(columns={"prediction": "power_kw"})
        return df
