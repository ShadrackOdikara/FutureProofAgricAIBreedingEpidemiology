#!/bin/env python
import os
import json
import ollama
import numpy as np
import pandas as pd
from numpy.linalg import norm

"""We have cities with varying temperature and humidity levels, which affect or impact the occurrence of different 
pathogens that cause diseases. In some cities, these parameters might not be ideal for a particular pathogen, making 
it less likely to occur, while in others, the conditions are highly likely and ideal.

In our code, the ideal situation is scored as binary, either 0 or 1. A score of 0 indicates that the current climate 
conditions do not meet the ideal situation for the proliferation of a pathogen, while a score of 1 means the ideal 
conditions are met. Therefore, for the three parameters we can measure—temperature, humidity, and wind speed—each is 
scored as either 1 or 0. Cumulatively, the total score should be 3 if all optimum conditions are met."""

class EmbeddingHandler:
    """Class to handle embedding storage and retrieval."""
    
    def __init__(self, directory="embeddings"):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_embeddings(self, filename, embeddings):
        """Save embeddings to a JSON file."""
        with open(f"{self.directory}/{filename}.json", "w") as f:
            json.dump(embeddings, f)

    def load_embeddings(self, filename):
        """Load embeddings from a JSON file."""
        filepath = f"{self.directory}/{filename}.json"
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r") as f:
            return json.load(f)

    def get_embeddings(self, filename, modelname, chunks):
        """Retrieve or compute embeddings."""
        if (embeddings := self.load_embeddings(filename)) is not False:
            return embeddings
        embeddings = [
            ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
            for chunk in chunks
        ]
        self.save_embeddings(filename, embeddings)
        return embeddings

class SimilarityFinder:
    """Class to handle similarity searches between embeddings."""
    
    @staticmethod
    def find_most_similar(needle, haystack):
        """Find most similar embeddings."""
        needle_norm = norm(needle)
        similarity_scores = [
            np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
        ]
        return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

class RiskVectorHandler:
    """Class to handle disease risk vector expansion and scoring."""
    
    def __init__(self, cities_data, disease_data):
        self.cities_data = cities_data
        self.disease_data = disease_data

    def expand_risk_vector(self):
        """Expand risk vector in cities data."""
        disease_name_scoring = self.disease_data["Disease/Disorder"].tolist()
        risk_vector_expanded = pd.DataFrame(self.cities_data["Risk_Vector"].tolist(), columns=disease_name_scoring)
        self.cities_data = pd.concat([self.cities_data, risk_vector_expanded], axis=1)
        self.cities_data.drop(columns=['Risk_Vector'], inplace=True)

    def get_city_risk(self, users_city):
        """Get the risk vector for a specific city."""
        return self.cities_data.loc[self.cities_data['City'] == users_city]

class PotatoModel:
    """Class to handle potato model-related tasks."""
    
    def __init__(self, model_name="potato_Wizard_v59"):
        self.model_name = model_name

    def get_prompt_embedding(self, prompt):
        """Get embedding for a given prompt."""
        return ollama.embeddings(model=self.model_name, prompt=prompt)["embedding"]


    def generate_response(self, prompt, matched_lines, epi_info):
        """Generate response from the model based on the prompt and matched lines."""
        return ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content":"You are a top-rated plant breeder and agronomy service agent named Buba. Give optimum potato trait combinations for potato varieties amid climate change and disease pressure. with the knowledge that a risk determinant is a cropland based method of determining the level of risks associated with plant disease for associated areas. The higher the risk the higher the risk determinant. We have cities with varying temperature, humidity levels and wind gusts, which affect or impact the occurrence of different pathogens that cause diseases. In some cities, these parameters might not be ideal for a particular pathogen, making it less likely to occur, while in others, the conditions are highly likely and ideal.The likelyhood of a disease occuring is scored and the higher the score for each disease the highly likely is the disease or pathogen to occur in that city."
                    + ": Please refer to the fact that " + "\n".join([f": {line}" for line in matched_lines])
                    + f"\nEpidemiological data:\n{epi_info}",
                },
                {"role": "user", "content": prompt},
            ],
        )


if __name__ == "__main__":
    # To Initialize handlers
    embedding_handler = EmbeddingHandler()
    similarity_finder = SimilarityFinder()
    risk_vector_handler = RiskVectorHandler(cities_data, disease_data)
    potato_model = PotatoModel()

    # Retrieve or compute embeddings dependng on user input weather contains a city or not
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

    #response = potato_model.generate_response(prompt, matched_lines, SYSTEM_PROMPT, epi_info)

