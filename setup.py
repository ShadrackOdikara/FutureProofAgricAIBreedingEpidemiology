from setuptools import find_packages,setup

setup(
    name="FutureAgriAiBreedingModel",
    version="0.0.1",
    author="Shadrack Odikara and Meshack Emakunat",
    author_email="shadrack.imai@gmail.com",
    install_requires=["langchain",
                      "streamlit",
                      "ollama",
                      "torch",
                      "gspread",
                      "networkx",
                      "python-dotenv",
                      "scikit-learn",
                      "re",
                      "os",
                      "json",
                      "transformers",
                      "numpy",
                      "pandas",
                      "sentence-transformers",
                      "accelerate"],
    packages=find_packages())
