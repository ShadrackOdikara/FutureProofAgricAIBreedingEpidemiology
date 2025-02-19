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
from math import log
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
determinant = float(matched_lines[0].split("Risk Determinant: ")[1]) if matched_lines else 1
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
scoring_input = response["message"]["content"] #response.message.content #: "Your response content here"}}  # Example input
model_name = "./scoring_model/fine_tuned_model_with_classification_head"

    # Initialize the SDGApplication
app = SDGApplication(data_path, model_name)

    # Run the application and get the top match
#top_score = app.run(scoring_input)
app.run(scoring_input)

similarities = app.model.calculate_similarity(scoring_input, app.hypotheses)
    
top_match_score = app.extract_top_match_score(similarities)

    #_, _, score = best_match
    #score = score
    # Print the best match explicitly if needed
"""   
if best_match:
    hypothesis, label, score = best_match
    global_score = score
    print("\n\nExplicit Best Match:")
    print(f"Hypothesis: {hypothesis}")
    print(f"Label: {label}")
    print(f"Confidence Score: {score:.2f}%")
else:
    print("No matching hypotheses found.")
"""

#_, _, score = app.run(scoring_input)#score  # Example score, replace with the actual score value
r = log(determinant)/100  # Dynamically calculate r based on the determinant
sim_manager = SimulatorManager(score, n_steps=5, r=r)  # Pass r explicitly
sim_manager.simulate_future_steps()

# Expose objects

__all__ = ["merged_df", "potato_disease_data", "epi_risks", "city_info", "cities_data", "users_city", "epi_info", "disease_data", "response", "score","sim_manager", "r", "determinant", "top_match_score"]

