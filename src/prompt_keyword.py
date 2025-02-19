#!/bin/env python
import re
import pandas as pd

"""
    Implementing and integrating the RAGs by referencing the results from the Epidemiology model, we use keyword 
concepts to capture and extract key cities from the user's input. However, using the split() method creates a 
list of all the words, making it difficult to capture two- and three-word cities, as well as handle punctuation 
effectively. Therefore, we opt to use regular expressions and the n-gram method, where the n-gram approach 
captures all names in sequence, including multi-word cities of up to three words.
"""

class KeywordExtractor:
    """Class to handle keyword extraction from user input using n-grams."""

    @staticmethod
    def extract(prompt, keywords, ngram_range=(1, 3)):
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


class RiskParser:
    """Class to handle parsing of risk data."""

    @staticmethod
    def parse(epi_risks, users_city):
        """
        Parse epi_risks and extract lines matching the user's city.

        Args:
            epi_risks: List of strings representing risk data.
            users_city: City extracted from the user's input.

        Returns:
            List of risk lines corresponding to the user's city.
        """
        if users_city:
            return [line for line in epi_risks if f"City: {users_city}" in line]
        else:
            return []


class CityInfo:
    """Class to manage the extraction of city-specific information."""

    def __init__(self, keywords, epi_risks, cities_data):
        """
        Initialize with keywords, risk data, and city data.

        Args:
            keywords: Set of city keywords.
            epi_risks: List of strings representing risk data.
            cities_data: DataFrame containing city-related information.
        """
        self.keywords = keywords
        self.epi_risks = epi_risks
        self.cities_data = cities_data

    def get_info(self, prompt):
        """
        Extract city information from the user's prompt.

        Args:
            prompt: User's input text.

        Returns:
            Tuple containing city info string and list of matched lines.
        """
        users_city = KeywordExtractor.extract(prompt, self.keywords, ngram_range=(1, 3))
        matched_lines = RiskParser.parse(self.epi_risks, users_city)

        if users_city:
            # Filter cities_data based on users_city
            city_data = self.cities_data[self.cities_data['City'] == users_city] 
            epi_info = city_data.to_json(orient="records", lines=False) 
        else:
            epi_info = "{}"

        if users_city:
            city_info = "\n".join([f": {line}" for line in matched_lines])
        else:
            city_info = {}#"No specific city information was found. General insights are provided based on climate and disease data."
        
        return city_info, matched_lines, epi_info


# Example usage
if __name__ == "__main__":
    # Define the list of cities
    keywords = set(cities_data['City'])  # Example keywords
    


    # Example prompt
    prompt = input(">>> ")
    # Example file content (epi_risks)
    #epi_risks = ["City: Nairobi Risk Level: High", "City: Lima Risk Level: Medium", "City: Mombasa Risk Level: Low"]

    # Create an instance of CityInfo and retrieve city information
    city_info_manager = CityInfo(keywords, epi_risks, cities_data) 
    city_info, matched_lines, epi_info = city_info_manager.get_info(prompt)
    determinant = float(matched_lines[0].split("Risk Determinant: ")[1]) if matched_lines else 1
    # Output the results
    print("City Info:", city_info)
    print("Matched Lines:", matched_lines)
    print("Epidemiological Information:", epi_info)
