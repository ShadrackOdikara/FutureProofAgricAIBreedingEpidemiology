from .dat_load import DataLoader
from .dat_load import DataLoader, Preprocessor, WeatherAnalyzer, Predictor
from .dis_epi_net import DataPreparation, Helpers, SuitabilityCalculator, RiskVectorUpdater, GraphInitializer, RiskPropagation,RiskEvaluator
from .prompt_keyword import KeywordExtractor, RiskParser, CityInfo
from .transformerRAGs import EmbeddingHandler, SimilarityFinder, RiskVectorHandler, PotatoModel
from .sentenceTrasformerScoring import SDGModel, SDGDataLoader, SDGPrinter, SDGApplication
from .markov_predict import BlackScholesSimulator, SimulatorManager
import pandas as pd
import numpy as np
import networkx as nx
from numpy.linalg import det
# Initialize data loader
data_loader = DataLoader()

# Load data
potato_disease_data, agro_ecology_zones, soil_data = data_loader.load_csv_files()
data = data_loader.load_google_sheets()

# Preprocess data
agro_ecology_zones = Preprocessor.preprocess_agroecology_zones(agro_ecology_zones)
soil_data = Preprocessor.preprocess_soil_data(soil_data)

# Generate statistics and predictions
summary_stats = WeatherAnalyzer.generate_summary_statistics(data)
target_date = "10/1/2025 13:07:01"
combined_predictions = Predictor.generate_combined_predictions(data, target_date)

# Merge data
"""For predicted weather data ureplace the summary statisytics below with combined_predictions"""

merged_df = pd.merge(summary_stats.reset_index(), agro_ecology_zones, on='City', how='outer')
merged_df = pd.merge(merged_df, soil_data, on='City', how='outer')


cities_data, disease_data = DataPreparation.initialize_data(merged_df, potato_disease_data)

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


# Define the list of cities
#keywords = {"Nairobi", "Lima", "Mombasa"}  # Example keywords
keywords = set(cities_data['City'])
# Example prompt
prompt = input(">>> ")
    # Example file content (epi_risks)
    #epi_risks = ["City: Nairobi Risk Level: High", "City: Lima Risk Level: Medium", "City: Mombasa Risk Level: Low"]

    # Create an instance of CityInfo and retrieve city information
city_info_manager = CityInfo(keywords, epi_risks, cities_data)
#city_info_manager = CityInfo(keywords, epi_risks)
#city_info, matched_lines, users_city = city_info_manager.get_info(prompt)
city_info, matched_lines, epi_info = city_info_manager.get_info(prompt)
#city_info, matched_lines, epi_info = city_info_manager.get_info(prompt)
#epi_info = cities_data.to_json(orient="records", lines=False) if users_city else "{}"    
    # Output the results
print("City Info:", city_info)
print("Matched Lines:", matched_lines)



# Initialize handlers
embedding_handler = EmbeddingHandler()
similarity_finder = SimilarityFinder()
risk_vector_handler = RiskVectorHandler(cities_data, disease_data)
potato_model = PotatoModel()

# Retrieve or compute embeddings
if matched_lines:
    embeddings = embedding_handler.get_embeddings("epi_risks", "potato_Wizard_v59", matched_lines)
    prompt_embedding = potato_model.get_prompt_embedding(prompt)
    most_similar_chunks = similarity_finder.find_most_similar(prompt_embedding, embeddings)[:5]

    for score, idx in most_similar_chunks:
        print(score, matched_lines[idx])
else:
    embeddings = []
    print("Skipping embedding generation as no matched lines were found.")

# Expand risk vector and get city-specific data
risk_vector_handler.expand_risk_vector()
#row = risk_vector_handler.get_city_risk(users_city)

# Generate response from the model
#SYSTEM_PROMPT = PotatoModel.SYSTEM_PROMPT
response = potato_model.generate_response(prompt, matched_lines, epi_info)





data_path = './data/'
scoring_input = response  # Example input
model_name = "./scoring_model/fine_tuned_model_with_classification_head"

app = SDGApplication(data_path, model_name)
app.run(scoring_input)


# Expose objects

__all__ = ["merged_df", "potato_disease_data", "epi_risks", "city_info", "cities_data", "users_city", "epi_info", "disease_data", "response"]

