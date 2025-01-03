# Future Proofing Agric With AI for Plant Breeding and Epidemiology Disease Management
\\
Welcome to our AI based crop epidemiological model designed to predict the optimum \
trait combination for potato cultivation in five countries namely Kenya, Uganda, \
Peru, China and India. This is based on an ensemble of various AI models and an \
epidemiological model as a Retrieval Augumented Generation RAGs system. \
\
The AI model described above introduces a novel approach to simulating the dynamics \
of crop disease outbreaks through a "country-blocked" network, capturing the complexities \
of agricultural trade, environmental conditions, and regional interactions. \
\
The model represents agricultural regions (e.g., cities or farming hubs) as nodes in a \
graph, with edges connecting them based on trade routes, shared climate zones, or geographical \
proximity. Each node is assigned attributes like crop type, disease susceptibility interms \
f cumulative score for each and every disease. \
\\
## How To Install 
\
### Download the model by git cloning the reposotory
\
Open the  ./AI_models/potato_Wizard_v59 \
Go to the file named Modelfiles \
Edit the path to your curent path e.g change ./home/shadrack/Documents/CIPotatoe \
To ./Your/path/potato_Wizard_v59/potato_Wizard_v59.gguf \
\
### Install ollama to your computer
Install Ollama on macOS: Use the command 
...bash 
curl https://ollama.ai/install.sh | sh. 
...\
Install Ollama on Linux: Use the command 
...bash
curl -fsSL https://ollama.ai/install.sh | bash. 
...\
Install Ollama on Windows: Download the installer from Ollama's official download page, run the installer, and follow the prompts. \
Install Ollama as a standalone CLI on Windows: Use the ollama-windows-amd64.zip zip file. \

Go to terminal and enter the command \
...bash
ollama create potato_Wizard_v59 -f Modelfile
...
\
wait until the creation is complete

## To use the model
Enter the command \
...bash
ollama run potato_Wizard_v59
...
\
You will get the 
...bash
>>> 
...
\
sign. Now you can continue interacting with the ai \

Type "who are you"


 