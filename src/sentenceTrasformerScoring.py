#!/bin/env python
import torch
from transformers import MPNetTokenizer, MPNetForSequenceClassification
import pandas as pd

class SDGModel:
    def __init__(self, model_name):
        """Initialize the tokenizer and model."""
        self.tokenizer = MPNetTokenizer.from_pretrained(model_name)
        self.model = MPNetForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Multi-class classification

    def calculate_similarity(self, response, hypotheses):
        """Calculate similarity between the response and hypotheses."""
        similarities = []
        for hypothesis in hypotheses:
            inputs = self.tokenizer(
                response, hypothesis, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            label_id = torch.argmax(probabilities, dim=-1).item()
            label = ["entailment", "contradiction", "neutral"][label_id]
            confidence_score = probabilities[0][label_id].item() * 100  # Convert to percentage

            similarities.append((hypothesis, label, confidence_score))

        # Sort by confidence and return the top matches
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
        return similarities

    def find_best_hypothesis(self, response, hypotheses, threshold=20.0):
        """Find the best matching hypothesis above a given threshold."""
        best_hypotheses = []

        for hypothesis in hypotheses:
            inputs = self.tokenizer(
                response, hypothesis, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            label_id = torch.argmax(probabilities, dim=-1).item()
            label = ["neutral", "contradiction", "entailment"][label_id]
            confidence_score = probabilities[0][label_id].item() * 100  # Convert to percentage

            # Only include hypotheses with confidence scores above the threshold
            if confidence_score >= threshold:
                best_hypotheses.append((hypothesis, label, confidence_score))

        # Sort by confidence score in descending order and select top matches
        best_hypotheses = sorted(best_hypotheses, key=lambda x: x[2], reverse=True)[:3]

        return best_hypotheses if best_hypotheses else [(None, "neutral", 0.0)]

class SDGDataLoader:
    @staticmethod
    def load_hypotheses(data_path):
        """Load SDG targets (hypotheses) from a file."""
        with open(data_path + 'SDGTargets.txt', 'r') as f:
            hypothesis_candidates = pd.read_table(f)
        return hypothesis_candidates['hypothesis'].tolist()

class SDGPrinter:
    @staticmethod
    def print_top_matching_hypotheses(response, similarities):
        """Print the top matching hypotheses with confidence scores."""
        print("Top Matching Hypotheses:")
        for hypothesis, label, score in similarities[:3]:  # Top 3 matches
            print("\n\n")
            print(response)
            print("\n")
            print(f"Your query aligns to global standard development goal : {hypothesis}")
            print(f"by : {label}")
            print(f"with a confidence score: {score:.2f}%\n")

class SDGApplication:
    def __init__(self, data_path, model_name):
        self.data_path = data_path
        self.model = SDGModel(model_name)
        self.hypotheses = SDGDataLoader.load_hypotheses(data_path)

    def run(self, scoring_input):
        """Run the application process."""
        generated_response = str(scoring_input)
        similarities = self.model.calculate_similarity(generated_response, self.hypotheses)
        SDGPrinter.print_top_matching_hypotheses(scoring_input, similarities)

    def extract_top_match_score(self, similarities):
        """Extract the top match score from the similarities."""
        if similarities:
            return similarities[0][2]
        return 0.0



# Example usage
if __name__ == "__main__":
    data_path = './data/'
    scoring_input = response.message.content #{"message": {"content": "Your response content here"}}  # Example input
    model_name = "./scoring_model/fine_tuned_model_with_classification_head"

    # Initialize the SDGApplication
    app = SDGApplication(data_path, model_name)

    # Run the application and get the top match

    app.run(scoring_input)

    similarities = app.model.calculate_similarity(scoring_input, app.hypotheses)
    
    top_match_score = app.extract_top_match_score(similarities)
    
#_, _, score = best_match
    #score = score
    # Print the best match explicitly if needed
    
#    if best_match:
#        hypothesis, label, score = best_match
#        global_score = score
#   else:
#        print("No matching hypotheses found.")
    

