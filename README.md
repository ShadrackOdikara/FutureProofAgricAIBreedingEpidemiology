Welcome to our AI based crop epidemiological model designed to predict the optimum trait combination 
for potato cultivation in five countries namely Kenya, Uganda, Peru, China and India. This is based on an ensemble of various AI 
models and an epidemiological model as a Retrieval Augumented Generation RAGs system.

The AI model described above introduces a novel approach to simulating the dynamics of crop disease outbreaks 
through a "country-blocked" network, capturing the complexities of agricultural trade, environmental conditions, 
and regional interactions.

The model represents agricultural regions (e.g., cities or farming hubs) as nodes in a graph, with edges connecting 
them based on trade routes, shared climate zones, or geographical proximity. Each node is assigned attributes 
like crop type, disease susceptibility interms of cumulative score for each and every disease.

To install 

Ensure you have python 3 installed in your system

use

Git clone the repository to the folder of your choice

pip install -r requirements.txt

=========================================================================================================
=========================================================================================================
To use the raw trained llm use ollama

Install Ollama on macOS: Use the command curl https://ollama.ai/install.sh | sh. 
Install Ollama on Linux: Use the command curl -fsSL https://ollama.ai/install.sh | bash. 
Install Ollama on Windows: Download the installer from Ollama's official download page, run the installer, and follow the prompts. 
Install Ollama as a standalone CLI on Windows: Use the ollama-windows-amd64.zip zip file. 

Open the  ./AI_models/potato_Wizard_v59
go to Model files 
Edit the path to youe curent path  ed change ./home/shadrack/Documents/CIPotatoe
to ./Your/path/potato_Wizard_v59/potato_Wizard_v59.gguf

go to terminal and enter the command 

ollama create potato_Wizard_v59 -f Modelfile

wait until the creation is complete

To use 
Enter the command 

ollama run potato_Wizard_v59

You will get the >>> sign 
and now you can continue interacting with the ai

Type "who are you"


 