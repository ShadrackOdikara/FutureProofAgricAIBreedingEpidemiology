#!/bin/env python
import numpy as np
import pandas as pd
import networkx as nx
from numpy.linalg import det
"""
    The epidemiological model desinged to estimate the prevalence and the spread of potato disease
across potato growing regions; fcusing on the cimatic conditions favoring disease causing pathogen
to proliferate in regions and the risk factor that other parameters such as trade routes, seed storage
seed handling, the alternative crop that pest and pathogens could hibernate to. All this risk factors a
aare captured into the model whereby the collective factors in a region/designated node are encompased as a vector unit
collectively ths vector unit is the risk factor of that node and the movement of seed and people
from one node to another posses the risk of spread of the disease. The disease can also be moved by 
animals wind direction floods and other natural phenomenas that are caused by climate change
The moment the risk vector move or transitions to another node there is some transformation in the vector risk
favtor  and thus making the vector to transform according to the influence of the transforming function/vector and in this case
it is the vector unit of risk assocated with movement such as trade routes either passing through disease infested 
areas or seed being poorly handled as it is moved from one farmer to another in the market place. this vector transform the
node vector that is moving and the vector that reaches the next node will be different or transformed accordingly.
Now the host node also has its risk vector and once the new vector come sin the nodes vector becomes a matrix encompassed 
by these two vectors

"""
class DataPreparation:
    @staticmethod
    def initialize_data(merged_df, PotatoDiseaseData):
        return merged_df, PotatoDiseaseData

class Helpers:
    @staticmethod
    def parse_range(range_str):
        """Parse string ranges (e.g., '20-25°C') into tuples of floats."""
        return tuple(map(float, range_str.replace("°C", "").replace("%", "").split('-')))

class SuitabilityCalculator:
    @staticmethod
    def calculate_suitability(city, disease_data):
        suitability_scores = []
        pH_min = city.get('pH Min', np.nan)
        pH_max = city.get('pH Max', np.nan)
        if pd.isna(pH_min) or pd.isna(pH_max) or pH_min > pH_max:
            pH_min, pH_max = 9999, 9999

        for _, disease_row in disease_data.iterrows():
            disease_name = disease_row['Disease/Disorder']
            temp_range = Helpers.parse_range(disease_row['Ideal Temperature Range (°C)'])
            pH_range = Helpers.parse_range(disease_row['Ideal pH Range'])
            humidity_range = Helpers.parse_range(disease_row['Ideal Humidity Range'])

            temp_suitability = int(temp_range[0] <= city['Mean Temperature'] <= temp_range[1])
            pH_suitability = int(not (pH_min > pH_range[0] or pH_max < pH_range[1]))
            humidity_suitability = int(humidity_range[0] <= city['Mean Humidity'] <= humidity_range[1])
            wind_suitability = 1

            total_score = temp_suitability + pH_suitability + humidity_suitability + wind_suitability
            suitability_scores.append([disease_name, total_score, city['Mean Temperature'], city['Mean Humidity'], city['Mean Wind Speed']])

        return suitability_scores

class RiskVectorUpdater:
    @staticmethod
    def update_risk_vectors(cities_data, disease_data):
        cities_data["Risk_Vector"] = [None] * len(cities_data)
        for index, city in cities_data.iterrows():
            suitability_scores = SuitabilityCalculator.calculate_suitability(city, disease_data)
            cities_data.at[index, "Risk_Vector"] = [score for _, score, _, _, _ in suitability_scores]
        return cities_data

class GraphInitializer:
    @staticmethod
    def initialize_graph(cities_data):
        G = nx.Graph()
        unique_zones = list(set(cities_data["Agroecological Zone"]))
        zone_mapping = {zone: idx for idx, zone in enumerate(unique_zones)}

        for _, city in cities_data.iterrows():
            G.add_node(
                city["City"],
                country=city["Country"],
                zone=city["Agroecological Zone"],
                zone_id=zone_mapping[city["Agroecological Zone"]],
                risk_vector=np.array(city["Risk_Vector"]),
            )

        city_names = [city["City"] for _, city in cities_data.iterrows()]
        for i, city1 in enumerate(city_names):
            for city2 in city_names[i + 1:]:
                edge_vector = np.random.uniform(0.1, 0.5, size=5)
                G.add_edge(city1, city2, edge_vector=edge_vector)

        return G, city_names

class RiskPropagation:
    @staticmethod
    def adjust_edge_vector(city_matrix, edge_vector):
        matrix_cols = city_matrix.shape[1] if len(city_matrix.shape) > 1 else len(city_matrix)
        edge_len = len(edge_vector)

        if edge_len < matrix_cols:
            edge_vector = np.pad(edge_vector, (0, matrix_cols - edge_len), 'constant')
        elif edge_len > matrix_cols:
            edge_vector = edge_vector[:matrix_cols]

        return edge_vector

    @staticmethod
    def calculate_transformed_vector(city_matrix, edge_vector):
        adjusted_edge_vector = RiskPropagation.adjust_edge_vector(city_matrix, edge_vector)
        return np.dot(city_matrix, adjusted_edge_vector)

    @staticmethod
    def gram_determinant(vectors):
        gram_matrix = np.dot(vectors, vectors.T)
        return det(gram_matrix)

    @staticmethod
    def propagate_risk_vectors(graph, target_city):
        incoming_transformed_vectors = []
        city_matrix = graph.nodes[target_city].get("risk_vector", np.eye(5))

        for neighbor in graph.neighbors(target_city):
            edge_vector = graph[neighbor][target_city]["edge_vector"]
            transformed_vector = RiskPropagation.calculate_transformed_vector(city_matrix, edge_vector)
            incoming_transformed_vectors.append(transformed_vector)

        if len(incoming_transformed_vectors) == 0:
            return None

        incoming_transformed_vectors = np.array(incoming_transformed_vectors)
        if len(incoming_transformed_vectors.shape) == 1:
            incoming_transformed_vectors = incoming_transformed_vectors[np.newaxis, :]

        if incoming_transformed_vectors.shape[0] >= incoming_transformed_vectors.shape[1]:
            incoming_transformed_vectors = incoming_transformed_vectors[:incoming_transformed_vectors.shape[1], :]

        return RiskPropagation.gram_determinant(incoming_transformed_vectors)

class RiskEvaluator:
    @staticmethod
    def evaluate_risks(graph, city_names):
        city_risk_determinants = {}
        for city in city_names:
            determinant = RiskPropagation.propagate_risk_vectors(graph, city)
            city_risk_determinants[city] = determinant
        return city_risk_determinants

# Main Execution
if __name__ == "__main__":
    # Initialize data
    cities_data, disease_data = DataPreparation.initialize_data(merged_df, PotatoDiseaseData)

    # Update risk vectors
    cities_data = RiskVectorUpdater.update_risk_vectors(cities_data, disease_data)

    # Initialize graph
    G, city_names = GraphInitializer.initialize_graph(cities_data)

    # Evaluate risks
    city_risk_determinants = RiskEvaluator.evaluate_risks(G, city_names)

    # Display risk determinants
    epi_risks = [
        f"City: {city}, Risk Determinant: {determinant}"
        for city, determinant in city_risk_determinants.items()
    ]
    for risk in epi_risks:
        print(risk)
