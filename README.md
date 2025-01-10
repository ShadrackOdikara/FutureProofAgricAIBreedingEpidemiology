# Future Proofing Agriculture with AI for Plant Breeding and Epidemiological Disease Management

Welcome to our AI-based crop epidemiological model, designed to predict the optimum trait combination for potato cultivation in five countries: Kenya, Uganda, Peru, China, and India. This model leverages an ensemble of various AI systems and an epidemiological model, functioning as a Retrieval-Augmented Generation (RAG) system.

The AI model introduces a novel approach to simulating the dynamics of crop disease outbreaks through a "country-blocked" network, capturing the complexities of agricultural trade, environmental conditions, and regional interactions.

The model represents agricultural regions (e.g., cities or farming hubs) as nodes in a graph, with edges connecting them based on trade routes, shared climate zones, or geographical proximity. Each node is assigned attributes such as crop type and disease susceptibility, represented as a cumulative score for each disease.


![The description of our model by prompting it to tell us about itself](/pictitures/Model%20Description.jpg)

---

## How to Install 

![Offline mode installion of the text generation model and usage for offline inference](/pictitures/installation%20instructions.jpg)

### Step 1: Download the Model 

#### Option 1: Download the Model in GGUF Format

File size is smaller

```bash
CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=1A0cRoxW0BiB7oHZfRNWSQ2UgOhEbOkeW" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=1A0cRoxW0BiB7oHZfRNWSQ2UgOhEbOkeW" -o potato_Wizard_v59.zip
```
or download directly from goggle drive

[Download from google drive](https://drive.google.com/file/d/1A0cRoxW0BiB7oHZfRNWSQ2UgOhEbOkeW/view?usp=drive_link)


#### Option 2: Download the the Full Models Combined

File size is large

```bash
CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${19cQGtvHRcTn4RRplXXiPuM5PJuwV3yUs}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${19cQGtvHRcTn4RRplXXiPuM5PJuwV3yUs}" -o ${AI_models.zip}
```
or download directly from goggle drive

[Download from google drive](https://drive.google.com/file/d/19cQGtvHRcTn4RRplXXiPuM5PJuwV3yUs/view?usp=drive_link)
### Step 2: Download the Scoring Model

```bash
CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${1o6qu2oUe9V0yattIyqCZ0P9aK4obO9Vt}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${1o6qu2oUe9V0yattIyqCZ0P9aK4obO9Vt}" -o ${scoring_model.zip}
```
or download directly from goggle drive

[Download from google drive](https://drive.google.com/file/d/1o6qu2oUe9V0yattIyqCZ0P9aK4obO9Vt/view?usp=drive_link)

### Step 3: Download and Install Ollama

Install Ollama on macOS: Use the command  
```bash 
curl https://ollama.ai/install.sh | sh. 
```

Install Ollama on Linux: Use the command  

```bash
curl -fsSL https://ollama.ai/install.sh | bash. 
```

Install Ollama on Windows: Download the installer from Ollama's official download page, run the installer, and follow the prompts. 
Install Ollama as a standalone CLI on Windows: Use the ollama-windows-amd64.zip zip file. 


### Step 4: Download the EFPR from the git hub repository by cloning

```bash
git clone https://github.com/ShadrackOdikara/FutureProofAgricAIBreedingEpidemiology.git
```


install dependencies by entering the command 

```bash
pip install -r requirements.txt
```
 

## How to Use the Model  

![The results output of the ensemble model after promting about which potato variety would be ideal to cultivate in Njoro Kenya](/pictitures/Example%20usage.jpg)

On the first line of Modelfile edit the path to reflect your computer path, where you are installing the model  

Go to terminal and enter the command   
```bash
ollama create potato_Wizard_v59 -f ./potato_Wizard_v59/Modelfile
```

wait until the creation is complete  

### To use the model

![Model usage output](/pictitures/Example%20two%20Model%20output.jpg)

Enter the CLI commands

```bash
python potato_WizardEFRP.py
```

or in a raw form while streaming

```bash
ollama run potato_Wizard_v59
```

You will get the 
```bash
>>> 
```

sign. Now you can continue interacting with the AI offline mode 

### Regions Currently Covered In The Epidemiological Modelling That You Can Query In the AI


'Nairobi', 'Lima', 'Beijing', 'New Delhi', 'Kampala','Ol Kalou', 'Engineer', 
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
            'Rwashamaire','Rubanda', 'Ikumba'


Type "who are you"

or "I want to cultivate potaoes in Chivay. Kindly suggest an ideal variety i should cultivate"


 