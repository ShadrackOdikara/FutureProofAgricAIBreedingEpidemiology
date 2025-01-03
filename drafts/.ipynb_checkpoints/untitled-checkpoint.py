import os
import re
import json
import torch
import gspread
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from numpy.linalg import norm, det
from sklearn.linear_model import LinearRegression
from oauth2client.service_account import ServiceAccountCredentials

# GPU Initialization
def initialize_gpu():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def authorize_google_sheets(credentials_file, scope):
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)
    return client

def fetch_google_sheet_data(client, sheet_name):
    sheet = client.open(sheet_name).sheet1
    data = sheet.get_all_records()
    return data

def save_data_locally(data, local_data_file):
    with open(local_data_file, "w") as d:
        json.dump(data, d, indent=4)

def load_local_data(local_data_file):
    with open(local_data_file, "r") as r:
        data = json.load(r)
    return data

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def preprocess_dataframe(data):
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M:%S")
    df.sort_values('Date', inplace=True)
    return df

def calculate_mean_last_intervals_per_city(df, column, intervals=7):
    return df.groupby('City')[column].apply(lambda group: group.tail(intervals).mean())

def merge_datasets(results, AgroEcologyZones):
    return pd.merge(results, AgroEcologyZones, on='City', how='outer')

def prepare_predictions(df, features, future_date):
    predictions = {feature: {} for feature in features}
    
    for city in df['City'].unique():
        city_df = df[df['City'] == city]
        X = city_df[['Timestamp']]
        models = {}

        for feature in features:
            y = city_df[feature]
            model = LinearRegression()
            model.fit(X, y)
            models[feature] = model

        future_timestamp = future_date.timestamp()
        for feature, model in models.items():
            predictions[feature][city] = model.predict([[future_timestamp]])[0]

    return pd.DataFrame(predictions).rename_axis('City')

def calculate_suitability(city, disease_data):
    suitability_scores = []

    for _, disease_row in disease_data.iterrows():
        disease_name = disease_row['Disease/Disorder']
        temp_range = parse_range(disease_row['Ideal Temperature Range (\u00b0C)'])
        humidity_range = parse_range(disease_row['Ideal Humidity Range'])
        
        temp_suitability = int(temp_range[0] <= city['Mean Temperature'] <= temp_range[1])
        humidity_suitability = int(humidity_range[0] <= city['Mean Humidity'] <= humidity_range[1])
        wind_suitability = 1
        
        total_score = temp_suitability + humidity_suitability + wind_suitability
        suitability_scores.append([disease_name, total_score])

    return suitability_scores

def parse_range(range_str):
    return tuple(map(float, range_str.replace("\u00b0C", "").replace("%", "").split('-')))

def initialize_graph(cities_data):
    G = nx.Graph()
    unique_zones = list(set(cities_data["Agroecological Zone"]))
    zone_mapping = {zone: idx for idx, zone in enumerate(unique_zones)}

    for _, city in cities_data.iterrows():
        G.add_node(city["City"], country=city["Country"], zone=city["Agroecological Zone"],
                   zone_id=zone_mapping[city["Agroecological Zone"]],
                   risk_vector=np.array(city.get("Risk_Vector", np.eye(5))))

    return G

def add_graph_edges(graph, city_names):
    for i, city1 in enumerate(city_names):
        for city2 in city_names[i + 1:]:
            edge_vector = np.random.uniform(0.1, 0.5, size=5)
            graph.add_edge(city1, city2, edge_vector=edge_vector)

def propagate_risk_vectors(graph, target_city):
    incoming_transformed_vectors = []
    city_matrix = graph.nodes[target_city].get("risk_vector", np.eye(5))

    for neighbor in graph.neighbors(target_city):
        edge_vector = graph[neighbor][target_city]["edge_vector"]
        transformed_vector = calculate_transformed_vector(city_matrix, edge_vector)
        incoming_transformed_vectors.append(transformed_vector)

    if not incoming_transformed_vectors:
        return None

    incoming_transformed_vectors = np.array(incoming_transformed_vectors)
    if incoming_transformed_vectors.shape[0] >= incoming_transformed_vectors.shape[1]:
        incoming_transformed_vectors = incoming_transformed_vectors[:incoming_transformed_vectors.shape[1], :]

    return gram_determinant(incoming_transformed_vectors)

def calculate_transformed_vector(city_matrix, edge_vector):
    adjusted_edge_vector = adjust_edge_vector(city_matrix, edge_vector)
    return np.dot(city_matrix, adjusted_edge_vector)

def adjust_edge_vector(city_matrix, edge_vector):
    matrix_cols = city_matrix.shape[1] if len(city_matrix.shape) > 1 else len(city_matrix)
    edge_len = len(edge_vector)

    if edge_len < matrix_cols:
        edge_vector = np.pad(edge_vector, (0, matrix_cols - edge_len), 'constant')
    elif edge_len > matrix_cols:
        edge_vector = edge_vector[:matrix_cols]

    return edge_vector

def gram_determinant(vectors):
    gram_matrix = np.dot(vectors, vectors.T)
    return det(gram_matrix)

def extract_keyword(prompt, keywords, ngram_range=(1, 3)):
    prompt_words = re.findall(r'\b\w+\b', prompt.title())
    all_ngrams = [
        ' '.join(prompt_words[i:i + n]) for n in range(ngram_range[0], ngram_range[1] + 1)
        for i in range(len(prompt_words) - n + 1)
    ]
    for ngram in all_ngrams:
        if ngram in keywords:
            return ngram
    return None

def save_embeddings(filename, embeddings):
    os.makedirs("embeddings", exist_ok=True)
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    filepath = f"embeddings/{filename}.json"
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r") as f:
        return json.load(f)
