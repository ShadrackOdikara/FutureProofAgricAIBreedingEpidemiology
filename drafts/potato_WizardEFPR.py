#!/bin/env python
"""
    We import packages that will be used to run the model and the RAGs system. This packages are for RAGs pipeline intergrating with ollama.
We also initialize that the model should run on a gpu should it find any gpu in the system.
"""

import os
import re
import json
import torch
import ollama
import gspread
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from numpy.linalg import norm, det
from sklearn.linear_model import LinearRegression
from oauth2client.service_account import ServiceAccountCredentials
#from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

#initialize the gpu should it be found in the system
torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "./data/"

""" We obtain climate data from the googlesheet database and if there is no internate access we obtain from the temporary directory
The weather data is real time data for cities within the countries of the project and the data taken is the temperature, humidity, wind,
precipitation; that includes rain and snow. All these conditions within the areas/cities contribute to the likelihood of pathogens to 
proliferate and cause disease. The climate data is obtained from the www.weatherData.com which updates daily to our google sheet

    This defines the scope of our external database where we are getting our climate data from that is the google sheets where we are storing
the weather data
"""

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Path to the credentials JSON file
credentials_file = data_path +'croplandepidemiology-92208cf30b77.json'

# Path to the local data file
local_data_file = data_path + "DATA.json"

"""Here we know that many users might not have the ability or resources to access the app online. Therefore we create a 
provision such that the last updated weater data will be stored locally in the users device and will be used in case the
device cannot be logged online and thus will only be used offline.
"""

try:
    # Load credentials and authorize the client
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    # Open the sheet by name
    sheet = client.open("Cropland_Epidemiology_Weather_Data").sheet1

    # Fetch all data
    data = sheet.get_all_records()

    # Save the data to a local file
    with open(local_data_file, "w") as d:
        json.dump(data, d, indent=4)

    print("Data successfully fetched from Google Sheets and saved locally.")

except Exception as e:
    print(f"Failed to access Google Sheets: {e}")
    print("Attempting to load data from the local file.")

    try:
        # Read the data from the local file
        with open(local_data_file, "r") as r:
            data = json.load(r)
            print("Data successfully loaded from the local file.")
    except FileNotFoundError:
        print("Local data file not found. No data available.")
        data = []

# Print the loaded data (optional)
#print(data)
"""Read the files to be used in the prediction, the potatoDiseaseprofile gives the ideal situations in term of humidity
temperature, soil ph that are ideal for the various potato pathogen to proliferate and cause disease. This is given in range
of temperature, humidity and soil ph that they wuld typically exist in"""

with open(data_path + 'potatoDiseaseprofile.csv', 'r') as potato_file:
    PotatoDiseaseData = pd.read_csv(potato_file)

with open(data_path + 'EpidemiologyAgroecologicalZones.csv', 'r') as zones_file:
    AgroEcologyZones = pd.read_csv(zones_file)

"""Clean the data and remove white spaces before and after"""
AgroEcologyZones = AgroEcologyZones.rename(columns=lambda x: x.strip())
AgroEcologyZones = AgroEcologyZones.map(lambda x: x.strip() if isinstance(x, str) else x)
data = pd.DataFrame(data)


# Load the dataset
data 
df = pd.DataFrame(data)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M:%S")

# Sort by Date
df.sort_values('Date', inplace=True)

# Function to calculate mean for the last 7 intervals per city
def calculate_mean_last_intervals_per_city(df, column, intervals=7):
    means = df.groupby('City')[column].apply(
        lambda group: group.tail(intervals).mean()
    )
    return means

#def calculate_mean_last_intervals_per_city(df, column, intervals=7):
#    means = df.groupby('City').apply(lambda group: group.tail(intervals)[column].mean())
#    return means

# Calculate mean Temperature, Wind Speed, Humidity, Precipitation, Rain, and Snow for the last 7 intervals per city
mean_temperature = calculate_mean_last_intervals_per_city(df, 'Temperature')
mean_wind_speed = calculate_mean_last_intervals_per_city(df, 'Wind Speed')
mean_humidity = calculate_mean_last_intervals_per_city(df, 'Humidity')
mean_precipitation = calculate_mean_last_intervals_per_city(df, 'Precipitation')
mean_rain = calculate_mean_last_intervals_per_city(df, 'Rain')
mean_snow = calculate_mean_last_intervals_per_city(df, 'Snow')

# Tabulate the results into a DataFrame
results = pd.DataFrame({
    'Mean Temperature': mean_temperature,
    'Mean Wind Speed': mean_wind_speed,
    'Mean Humidity': mean_humidity,
    'Mean Precipitation': mean_precipitation,
    'Mean Rain': mean_rain,
    'Mean Snow': mean_snow
})

print("Results for the last 7 days/intervals per city:")
print(results)

""" We reset the index for the mean data per city to enable merging with the agro-ecological zone description data.
Next, we merge the two dataframes to create a single dataframe containing all the weather information along with the 
corresponding attributes of the area the weather data pertains to. These areas represent known potato growing regions 
for each country within the hackathon/project.
"""
results = results.reset_index()
merged_df = pd.merge(results, AgroEcologyZones, on='City', how='outer')

"""  
    Here we introduce linear regression to predict the future state of the climate parameters that will be used 
in the epidemiological model prediction. This will be based on users preferences on setting s future date in their
query.

These are
    - Future temperature states
    - Future humidity states
    - Future projected wind speeds in the areas
    - The precipitation including
    - The rain
    - The snow

The precipitations have been commmented out since there are areas with zeros but these will be computed by 
uncommenting them and introducing coercion.

    We use a for loop to iterate over the dependent features one at a time while computing the linear regression 
predicting future weather conditions.
"""
# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])
df['Timestamp'] = df['Date'].apply(lambda x: x.timestamp())

# Dictionaries to store predictions
predictions = {
    'Temperature': {},
    'Humidity': {},
    'Wind Speed': {},
    #'Precipitation': {},
    #'Rain': {},
    #'Snow': {}
}

# Iterate through each city
for city in df['City'].unique():
    city_df = df[df['City'] == city]
    X = city_df[['Timestamp']]
    
    # Train models for each feature
    models = {}
    for feature in ['Temperature', 
                    'Humidity', 
                    'Wind Speed', 
                    #'Precipitation', 
                    #'Rain', 
                    #'Snow'
                   ]:
        y = city_df[feature]
        model = LinearRegression()
        model.fit(X, y)
        models[feature] = model

    # Predict future values for 10/1/2025
    future_date = datetime.strptime("10/1/2025 13:07:01", "%d/%m/%Y %H:%M:%S")
    future_timestamp = future_date.timestamp()
    for feature, model in models.items():
        predictions[feature][city] = model.predict([[future_timestamp]])[0]

# Combine predictions into a single DataFrame
combined_predictions = pd.DataFrame(predictions)
combined_predictions.index.name = 'City'

# Print the combined DataFrame
print("\nPredicted values for 10/1/2025 per city:")
print(combined_predictions)

#print(hum_predictions_df)

#print(f"{city}: {temp:.2f} {hum:.2f}")
   


"""The epidemiological model desinged to estimate the prevalence and the spread of potato disease
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
by these two vectors"""

# Define cities data
cities_data = merged_df

# Define disease data
disease_data = PotatoDiseaseData

# Helper function to parse ranges
def parse_range(range_str):
    """Parse string ranges (e.g., '20-25°C') into tuples of floats."""
    return tuple(map(float, range_str.replace("°C", "").replace("%", "").split('-')))

# Suitability calculation function
def calculate_suitability_from_disease_data(city, disease_data):
    suitability_scores = []
    for _, disease_row in disease_data.iterrows():
        disease_name = disease_row['Disease/Disorder']
        temp_range = parse_range(disease_row['Ideal Temperature Range (°C)'])
        pH_range = parse_range(disease_row['Ideal pH Range'])
        humidity_range = parse_range(disease_row['Ideal Humidity Range'])

        # Calculate suitability
        temp_suitability = int(temp_range[0] <= city['Mean Temperature'] <= temp_range[1])
        #pH_suitability = int(pH_range[0] <= city['Soil_pH'] <= pH_range[1])
        humidity_suitability = int(humidity_range[0] <= city['Mean Humidity'] <= humidity_range[1])
        wind_suitability = 1  # Later on we will add wind speed criteria here

        # Calculate total suitability score
        total_score = temp_suitability + humidity_suitability + wind_suitability
        suitability_scores.append([disease_name, total_score, city['Mean Temperature'], city['Mean Humidity'], city['Mean Wind Speed']])

    return suitability_scores
    
cities_data["Risk_Vector"] = np.nan

# Ensure Risk_Vector column exists
cities_data["Risk_Vector"] = [None] * len(cities_data)

# Update Risk_Vector column
for index, city in cities_data.iterrows():
    suitability_scores = calculate_suitability_from_disease_data(city, disease_data)
    
    # Ensure it's stored as a list
    cities_data.at[index, "Risk_Vector"] = [score for _, score, _, _, _ in calculate_suitability_from_disease_data(city, disease_data)]

# Output updated DataFrame
print(cities_data[['City', 'Risk_Vector']])


# Graph initialization
G = nx.Graph()
unique_zones = list(set(cities_data["Agroecological Zone"]))

zone_mapping = {zone: idx for idx, zone in enumerate(unique_zones)}

# Add nodes with updated risk vectors
for _, city in cities_data.iterrows():
    G.add_node(
        city["City"],
        country=city["Country"],
        zone=city["Agroecological Zone"],
        zone_id=zone_mapping[city["Agroecological Zone"]],
        risk_vector=np.array(city["Risk_Vector"]),
    )

# Generate city names from DataFrame
city_names = [city["City"] for _, city in cities_data.iterrows()]

# Assign a unique vector to each edge
for i, city1 in enumerate(city_names):
    for city2 in city_names[i + 1:]:
        edge_vector = np.random.uniform(0.1, 0.5, size=5)  # 5-dimensional vector for each edge
        G.add_edge(city1, city2, edge_vector=edge_vector)


"""Function to calculate cosine similarity. Here create a function to Calculate the matrix - vector product.... 
that is the matrix in the node by the edge_vector in the bidirectional network from one city to another
These are Functions to calculate the matrix-vector product"""

def calculate_transformed_vector(risk_vector, edge_vector):
    """
    Compute the transformed vector by multiplying the city's matrix
    by the edge vector.
    """
    return np.dot(risk_vector, edge_vector)



# Use Gram determinant for non-square matrices
def gram_determinant(vectors):
    """Calculate the determinant of the Gram matrix formed by vectors."""
    gram_matrix = np.dot(vectors, vectors.T)
    return det(gram_matrix)

# Function to adjust edge vector dynamically
def adjust_edge_vector(city_matrix, edge_vector):
    """
    Adjust the edge vector to match the city's matrix column dimensions.
    If too short, pad with zeros. If too long, truncate.
    """
    matrix_cols = city_matrix.shape[1] if len(city_matrix.shape) > 1 else len(city_matrix)
    edge_len = len(edge_vector)
    
    if edge_len < matrix_cols:
        # Pad edge vector with zeros
        edge_vector = np.pad(edge_vector, (0, matrix_cols - edge_len), 'constant')
    elif edge_len > matrix_cols:
        # Truncate edge vector
        edge_vector = edge_vector[:matrix_cols]

    return edge_vector

# function for matrix-vector product
def calculate_transformed_vector(city_matrix, edge_vector):
    """
    Compute the transformed vector by multiplying the city's matrix
    by the edge vector. Adjust the edge vector dynamically if needed.
    """
    adjusted_edge_vector = adjust_edge_vector(city_matrix, edge_vector)
    return np.dot(city_matrix, adjusted_edge_vector)

# propagate_risk_vectors function
def propagate_risk_vectors(graph, target_city):
    """
    Propagate risk vectors through the network by combining edge vectors
    with city matrices, and calculate the Gram determinant.
    """
    incoming_transformed_vectors = []

    # Retrieve the target city's matrix
    city_matrix = graph.nodes[target_city].get("risk_vector", np.eye(5))  # Default to identity matrix if missing

    for neighbor in graph.neighbors(target_city):
        # Retrieve the edge vector
        edge_vector = graph[neighbor][target_city]["edge_vector"]

        # Adjust dimensions and transform the city's matrix with the edge vector
        transformed_vector = calculate_transformed_vector(city_matrix, edge_vector)

        # Append the transformed vector to incoming vectors
        incoming_transformed_vectors.append(transformed_vector)

    # Check if there are any incoming transformed vectors
    if len(incoming_transformed_vectors) == 0:
        return None  # No valid incoming vectors to process

    # Convert to a numpy array
    incoming_transformed_vectors = np.array(incoming_transformed_vectors)

    # Ensure the array is at least 2D
    if len(incoming_transformed_vectors.shape) == 1:
        incoming_transformed_vectors = incoming_transformed_vectors[np.newaxis, :]

    # If more rows than columns, truncate to square for determinant
    if incoming_transformed_vectors.shape[0] >= incoming_transformed_vectors.shape[1]:
        incoming_transformed_vectors = incoming_transformed_vectors[:incoming_transformed_vectors.shape[1], :]

    # Calculate Gram determinant
    determinant = gram_determinant(incoming_transformed_vectors)
    return determinant


# Evaluate risk for each city
city_risk_determinants = {}
for city in city_names:
    determinant = propagate_risk_vectors(G, city)
    city_risk_determinants[city] = determinant

# Display risk determinants
epi_risks = [
    f"City: {city}, Risk Determinant: {determinant}" 
    for city, determinant in city_risk_determinants.items()
]

"""
    Implementing and integrating the RAGs by referencing the results from the Epidemiology model, we use keyword 
concepts to capture and extract key cities from the user's input. However, using the split() method creates a 
list of all the words, making it difficult to capture two- and three-word cities, as well as handle punctuation 
effectively. Therefore, we opt to use regular expressions and the n-gram method, where the n-gram approach 
captures all names in sequence, including multi-word cities of up to three words.
"""

keywords = set(cities_data['City'])

"""
    An example of cities that the were extracted weater data and passed through the epidemiology workflow. These cities
can be expanded to include more cities as the scope of the project increases. This will be changed from the Google Apps Script
for extracting weather data from various cities

{'Nairobi', 'Lima', 'Beijing', 'New Delhi', 'Kampala','Ol Kalou', 'Engineer', 
            'Njabini','Meru', 'Maua', 'Timau','Iten', 'Kapsowar','Molo', 'Njoro', 'Nakuru', 
            'Kuresoi','Thika', 'Limuru', 'Githunguri','Bomet', 'Sotik', 'Mulot', 'Sunset Town',
            'Nyeri', 'Karatina','Kerugoya', 'Kutus','Eldoret', 'Turbo', 'Moi\'s Bridge','Kitale', 
            'Endebess','Kapenguria', 'Makutano','Bungoma', 'Webuye','Cuzco', 'Pisac', 'Urubamba',
            'Puno', 'Juliaca', 'Ayaviri','Huancavelica', 'Lircay', 'Pampas','Huancayo', 'Jauja', 
            'Tarma','Ayacucho', 'Huanta', 'San Miguel','Huaraz', 'Caraz', 'Chacas','Cajamarca', 
            'Celendín', 'Jaén','Arequipa', 'Chivay', 'Camaná','Abancay', 'Andahuaylas','Huaral', 
            'Huacho', 'Canta','Cerro de Pasco', 'Oxapampa','Trujillo', 'Otuzco', 'Huamachuco', 
            'Agra', 'Kannauj', 'Farrukhabad', 'Aligarh', 'Kolkata', 'Hooghly', 'Bardhaman', 
            'Cooch Behar', 'Patna', 'Nalanda', 'Muzaffarpur', 'Gaya','Jalandhar', 'Amritsar', 
            'Ludhiana', 'Hoshiarpur','Ahmedabad', 'Deesa', 'Gandhinagar', 'Banaskantha', 'Indore', 
            'Gwalior', 'Bhopal', 'Hoshangabad','Shimla', 'Solan', 'Mandi', 'Kullu','Dehradun', 
            'Nainital', 'Almora', 'Haldwani','Bengaluru', 'Hassan', 'Mysuru', 'Chikkamagaluru',
            'Guwahati', 'Jorhat', 'Tezpur', 'Dibrugarh','Ranchi', 'Dhanbad', 'Hazaribagh', 
            'Bokaro','Shillong', 'Tura', 'Jowai','Hohhot', 'Baotou', 'Chifeng','Harbin', 'Qiqihar', 
            'Mudanjiang','Lanzhou', 'Tianshui', 'Dingxi','Kunming', 'Dali', 'Lijiang','Chengdu', 
            'Mianyang', 'Dazhou','Guiyang', 'Anshun', 'Zunyi','Xian', 'Baoji', 'Yulin','Chongqing', 
            'Wanzhou', 'Yongchuan','Yinchuan', 'Shizuishan', 'Zhongwei','Urumqi', 'Kashgar', 'Korla', 
            'Xining', 'Golmud','Lhasa', 'Shigatse','Kabale', 'Katuna','Kisoro', 'Bunagana','Mbale', 
            'Bududa','Kapchorwa', 'Suam','Kasese', 'Hima','Fort Portal', 'Kijura', 'Bundibugyo', 
            'Nyahuka','Mbarara', 'Isingiro','Bushenyi', 'Ishaka','Rukungiri', 'Kanungu','Ntungamo', 
            'Rwashamaire','Rubanda', 'Ikumba'}
"""
def extract_keyword(prompt, keywords, ngram_range=(1, 3)):
    """
    Extract a matching keyword from the user's prompt using n-grams.

    Args:
        prompt: User's input text.
        keywords: Set of keywords to match.
        ngram_range: Tuple specifying the range of n-grams to generate (e.g., 1 to 3 words).

    Returns:
        The first matching keyword found in the prompt, or None if no match is found.
    """
    # Tokenize prompt into words
    prompt_words = re.findall(r'\b\w+\b', prompt.title())
    all_ngrams = []

    # Generate n-grams within the specified range
    for n in range(ngram_range[0], ngram_range[1] + 1):
        all_ngrams.extend([' '.join(prompt_words[i:i + n]) for i in range(len(prompt_words) - n + 1)])

    # Find and return the first matching keyword
    for ngram in all_ngrams:
        if ngram in keywords:
            return ngram
    return None

# Set the system prompt which will later be appended with the RAGs inputs
SYSTEM_PROMPT = """You are a top-rated plant breeder and agronomy service agent named Buba. 
Give optimum potato trait combinations for potato varieties amid climate change and disease pressure. 
with the knowledge that a risk determinant is a cropland based method of determining the level of risks 
associated with plant disease for associated areas. The higher the risk the higher the risk determinant. 

We have cities with varying temperature, humidity levels and wind gusts, which affect or impact the occurrence of different 
pathogens that cause diseases. In some cities, these parameters might not be ideal for a particular pathogen, making 
it less likely to occur, while in others, the conditions are highly likely and ideal.

The likelyhood of a disease occuring is scored and the higher the score for each disease the highly likely is the 
disease or pathogen to occur in that city. 
"""

prompt = input(">>> ")
#prompt = "Which is the best potato variety to cultivate in Mulot?"
users_city = extract_keyword(prompt, keywords, ngram_range=(1, 3))

def parse_file(epi_risks, users_city):
    """Parse epi_risks and extract lines matching the user's city."""
    risk_lines = [line for line in epi_risks if f"City: {users_city}" in line]
    return risk_lines

if users_city:
    matched_lines = parse_file(epi_risks, users_city)
    print("Matched Lines:", matched_lines)
else:
    matched_lines = []
    print("No matched lines since no city was identified.")
    
def save_embeddings(filename, embeddings):
    """Save embeddings to a JSON file."""
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    """Load embeddings from a JSON file."""
    filepath = f"embeddings/{filename}.json"
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    """Retrieve or compute embeddings."""
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    """Find most similar embeddings."""
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# Generate embeddings
embeddings = get_embeddings("epi_risks", "potato_Wizard_v59", matched_lines)


"""We have cities with varying temperature and humidity levels, which affect or impact the occurrence of different 
pathogens that cause diseases. In some cities, these parameters might not be ideal for a particular pathogen, making 
it less likely to occur, while in others, the conditions are highly likely and ideal.

In our code, the ideal situation is scored as binary, either 0 or 1. A score of 0 indicates that the current climate 
conditions do not meet the ideal situation for the proliferation of a pathogen, while a score of 1 means the ideal 
conditions are met. Therefore, for the three parameters we can measure—temperature, humidity, and wind speed—each is 
scored as either 1 or 0. Cumulatively, the total score should be 3 if all optimum conditions are met."""

disease_name_scoring = disease_data["Disease/Disorder"].tolist()
risk_vector_expanded = pd.DataFrame(cities_data["Risk_Vector"].tolist(), columns=disease_name_scoring)
cities_data = pd.concat([cities_data, risk_vector_expanded], axis=1)
print(cities_data.head())

cities_data.drop(columns=['Risk_Vector'], inplace=True)
row = cities_data.loc[cities_data['City'] == users_city]
epi_info = row.to_json(orient="records", lines=False)


# Prompt embedding and similarity search
#prompt = "Which is the best potato variety to cultivate in Molo?"
prompt_embedding = ollama.embeddings(model="potato_Wizard_v59", prompt=prompt)["embedding"]
most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

for score, idx in most_similar_chunks:
    print(score, matched_lines[idx])

response = ollama.chat(
    model="potato_Wizard_v59",
    #stream = True,
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + ": Please refer to the fact that " + "\n".join([f": {line}" for line in matched_lines]) + epi_info,
        },
        {"role": "user", "content": prompt},
    ],
)
#print("\n\n")
#print(response["message"]["content"])

scoring_input = response["message"]["content"]

"""
    We now use the scoring model to future proof the query and try an measure how it contribute to global set goals. 
Here we implement the scoring model based on sequence classification task which the MPNet model was fine tuned and 
modified to perform against the set of goal hypothesis

"""


with open(data_path + 'SDGTargets.txt', 'r') as f:
    hypothesis_candidates = pd.read_table(f)
    #print(hypothesis_candidates)

hypothesis_candidates = hypothesis_candidates['hypothesis'].tolist()


from transformers import MPNetTokenizer, MPNetForSequenceClassification
import torch

# Load the tokenizer and pre-trained model
model_name = "./scoring_model/fine_tuned_model_with_classification_head"  
tokenizer = MPNetTokenizer.from_pretrained(model_name)
model = MPNetForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Multi-class classification

# We get the input for classification from the output of the potatoWizard
generated_response = str(scoring_input)

##############################################################################
def calculate_similarity(response, hypotheses):
    similarities = []
    for hypothesis in hypotheses:
        inputs = tokenizer(response, hypothesis, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        label_id = torch.argmax(probabilities, dim=-1).item()
        label = ["entailment", "contradiction", "neutral"][label_id]
        confidence_score = probabilities[0][label_id].item() * 100  # Convert to percentage

        similarities.append((hypothesis, label, confidence_score))

    # Sort by confidence and return the top matches
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    return similarities

##############################################################################
def find_best_hypothesis(response, hypotheses, threshold=20.0):
    best_hypotheses = []

    for hypothesis in hypotheses:
        inputs = tokenizer(response, hypothesis, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        label_id = torch.argmax(probabilities, dim=-1).item()
        label = ["entailment", "contradiction", "neutral"][label_id]
        confidence_score = probabilities[0][label_id].item() * 100  # Convert to percentage

        # Only include hypotheses with confidence scores above the threshold
        if confidence_score >= threshold:
            best_hypotheses.append((hypothesis, label, confidence_score))

    # Sort by confidence score in descending order and select top matches
    best_hypotheses = sorted(best_hypotheses, key=lambda x: x[2], reverse=True)[:3]

    return best_hypotheses if best_hypotheses else [(None, "neutral", 0.0)]

##############################################################################
# Compute similarities
similarities = calculate_similarity(generated_response, hypothesis_candidates)

# Print the top 3 most similar hypotheses
print("Top Matching Hypotheses:")
for hypothesis, label, score in similarities[:3]:  # Top 3 matches
    print("\n\n")
    print(response["message"]["content"])
    print("\n\n")
    print(f"Your query aligns to global standard development goal : {hypothesis}")
    print(f"by : {label}")
    print(f"with a confidence score: {score:.2f}%\n")

##############################################################################



