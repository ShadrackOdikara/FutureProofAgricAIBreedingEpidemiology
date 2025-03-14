from setuptools import find_packages,setup

setup(
    name="FutureAgriAiBreedingModel",
    version="0.0.1",
    author="Shadrack Odikara and Meshack Emakunat",
    author_email="shadrack.imai@gmail.com",
    install_requires=["gspread==6.1.4",
    		      #"networkx==3.1",
    		      "mpmath==1.3.0",
                      "networkx==3.3",
                      "numpy==2.2.1",
                      "oauth2client==4.1.3",
                      "ollama==0.4.5",
                      "pandas==2.0.3",
                      "protobuf==5.29.2",
                      "scikit_learn==1.3.0",
                      #"setuptools==68.0.0",
                      "setuptools==65.5.0",
                      #"torch==2.2.2+cpu",
                      "torch==2.4.0",
                      #"transformers==4.47.0",
                      "transformers==4.46.2"],
    packages=find_packages())
