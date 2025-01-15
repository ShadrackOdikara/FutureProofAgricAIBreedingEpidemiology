#!/bin/env python
"""
    Data loader for the ensemble model desinged to facilitate offline running of the model by storing data and retriving it
while offline. Data include weather data from google sheets, custom potato disease dataset, Ah=gro ecological zones for five
countries and soil data and the estimated pH ranges
"""

import os
import re
import json
import torch
import gspread
import numpy as np
import pandas as pd
from datetime import datetime
from numpy.linalg import norm, det
from sklearn.linear_model import LinearRegression
from oauth2client.service_account import ServiceAccountCredentials

class DataLoader:
    def __init__(self, data_path="./data/", credentials_file="croplandepidemiology-92208cf30b77.json"):
        self.data_path = data_path
        self.credentials_file = os.path.join(data_path, credentials_file)
        self.local_data_file = os.path.join(data_path, "DATA.json")

    def load_google_sheets(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, scope)
            client = gspread.authorize(creds)
            sheet = client.open("Cropland_Epidemiology_Weather_Data").sheet1
            data = sheet.get_all_records()
            with open(self.local_data_file, "w") as d:
                json.dump(data, d, indent=4)
            print("Data successfully fetched from Google Sheets and saved locally.")
        except Exception as e:
            print(f"Failed to access Google Sheets: {e}")
            print("Attempting to load data from the local file.")
            try:
                with open(self.local_data_file, "r") as r:
                    data = json.load(r)
                    print("Data successfully loaded from the local file.")
            except FileNotFoundError:
                print("Local data file not found. No data available.")
                data = []
        return pd.DataFrame(data)

    def load_csv_files(self):
        potato_disease_data = pd.read_csv(os.path.join(self.data_path, 'potatoDiseaseprofile.csv'))
        agro_ecology_zones = pd.read_csv(os.path.join(self.data_path, 'EpidemiologyAgroecologicalZones.csv'))
        soil_data = pd.read_csv(os.path.join(self.data_path, 'soilData.csv'))
        return potato_disease_data, agro_ecology_zones, soil_data


class Preprocessor:
    @staticmethod
    def preprocess_agroecology_zones(agro_ecology_zones):
        agro_ecology_zones = agro_ecology_zones.rename(columns=lambda x: x.strip())
        agro_ecology_zones = agro_ecology_zones.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return agro_ecology_zones

    @staticmethod
    def preprocess_soil_data(soil_data):
        def parse_soilrange(range_str):
            return tuple(map(float, range_str.replace("–", "-").split("-")))

        soil_data[['pH Min', 'pH Max']] = soil_data['pH Range'].apply(parse_soilrange).apply(pd.Series)
        return soil_data


class WeatherAnalyzer:
    @staticmethod
    def preprocess_date_and_timestamp(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Timestamp'] = df['Date'].apply(lambda x: x.timestamp())
        return df

    @staticmethod
    def calculate_mean_last_intervals_per_city(df, column, intervals=7):
        return df.groupby('City')[column].apply(lambda group: group.tail(intervals).mean())

    @staticmethod
    def generate_summary_statistics(df):
        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M:%S")
        df.sort_values('Date', inplace=True)
        stats = {
            'Mean Temperature': WeatherAnalyzer.calculate_mean_last_intervals_per_city(df, 'Temperature'),
            'Mean Wind Speed': WeatherAnalyzer.calculate_mean_last_intervals_per_city(df, 'Wind Speed'),
            'Mean Humidity': WeatherAnalyzer.calculate_mean_last_intervals_per_city(df, 'Humidity'),
            'Mean Precipitation': WeatherAnalyzer.calculate_mean_last_intervals_per_city(df, 'Precipitation'),
            'Mean Rain': WeatherAnalyzer.calculate_mean_last_intervals_per_city(df, 'Rain'),
            'Mean Snow': WeatherAnalyzer.calculate_mean_last_intervals_per_city(df, 'Snow')
        }
        return pd.DataFrame(stats)


class Predictor:
    @staticmethod
    def train_models_for_city(city_df):
        features = ['Temperature', 'Humidity', 'Wind Speed']
        X = city_df[['Timestamp']]
        models = {}
        for feature in features:
            y = city_df[feature]
            model = LinearRegression()
            model.fit(X, y)
            models[feature] = model
        return models

    @staticmethod
    def predict_future_values(models, future_timestamp, city):
        predictions = {}
        for feature, model in models.items():
            predictions[feature] = model.predict([[future_timestamp]])[0]
        return {city: predictions}

    @staticmethod
    def generate_combined_predictions(df, target_date):
        df = WeatherAnalyzer.preprocess_date_and_timestamp(df)
        predictions = {'Temperature': {}, 'Humidity': {}, 'Wind Speed': {}}

        future_date = datetime.strptime(target_date, "%d/%m/%Y %H:%M:%S")
        future_timestamp = future_date.timestamp()

        for city in df['City'].unique():
            city_df = df[df['City'] == city]
            models = Predictor.train_models_for_city(city_df)
            city_predictions = Predictor.predict_future_values(models, future_timestamp, city)

            for feature, value in city_predictions[city].items():
                predictions[feature][city] = value

        combined_predictions = pd.DataFrame(predictions)
        combined_predictions.rename(columns={'Temperature': 'Mean Temperature', 'Humidity': 'Mean Humidity', 'Wind Speed': 'Mean Wind Speed'}, inplace=True)

        combined_predictions.index.name = 'City'
        return combined_predictions


# Main execution logic
if __name__ == "__main__":
    data_loader = DataLoader()
    data = data_loader.load_google_sheets()
    potato_disease_data, agro_ecology_zones, soil_data = data_loader.load_csv_files()

    agro_ecology_zones = Preprocessor.preprocess_agroecology_zones(agro_ecology_zones)
    soil_data = Preprocessor.preprocess_soil_data(soil_data)

    summary_stats = WeatherAnalyzer.generate_summary_statistics(data)
    target_date = "10/1/2025 13:07:01"

    combined_predictions = Predictor.generate_combined_predictions(data, target_date)
    print("\nPredicted values for 10/1/2025 per city:")
    print(combined_predictions)
    #combined_predictions.rename(columns={'Temperature': 'Mean Temperature', 'Humidity': 'Mean Humidity', 'Wind Speed': 'Mean Wind Speed'}, inplace=True)

    combined_predictions = combined_predictions.reset_index()

    """For predicted weather data ureplace the summary statisytics below with combined_predictions"""

    merged_df = pd.merge(summary_stats.reset_index(), agro_ecology_zones, on='City', how='outer')
    merged_df = pd.merge(merged_df, soil_data, on='City', how='outer')
    

    print("Summary Statistics:")
    print(summary_stats)
