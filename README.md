# Plant disease classification from images
## Project overview
This is my capstone project from my Data Science & AI Bootcamp (09.2025 – 01.2026, SPICED Academy x Neuefische). The goal of this project is to create a CNN model that would recognize ten common diseases of agricultural plants from pictures taken under field conditions (with emphasis on night shade plants – explanation follows below). It is a well-known problem that models trained on uniform lab images do not generalize well on real-life images. This project aims to overcome this obstacle by introducing hand-picked “wild” images into the training set and oversampling them during the training. The resulting model is deployed via Streamlit and can be seen here: https://plant-disease-classification-wild-generalization-tw2vcxfnkogcv.streamlit.app/.
## Dataset
The dataset deliberately focuses primarily on night shade plants (tomato, potato, bell pepper) to downsize the original dataset for the purposes of a capstone project. All the images are labeled corresponding to the plant disease they are showing.
The dataset consists of two parts:
1)	Plant Village dataset (see background information here: https://plantvillage.psu.edu/): ~20k lab images with a single leaf on grey background; obtained from here: https://www.kaggle.com/datasets/moazeldsokyx/plantvillage?resource=download
2)	Wild dataset: ~250 images collected manually from various trustworthy online sources – universities’ websites, PlantVillage app images, scientific publications or books, larger agricultural advice websites, websites of companies producing pesticides.
## Project structure
1)	Data import, preprocessing, train / validation / test split. 
2)	Baseline model (courtesy of my teammate Zabihullah Sherzad, github name: Sherzadd): custom CNN from scratch, trained on Plant Village images only. 
3)	Pretrained model using EfficientNetB0 pretrained network. Trained on a mixed dataset (0.3 Plant Village, 0.7 wild images).
4)	Test (accuracy) on wild images only.
## Results & Outlook
As of now, the accuracy on wild test images was increased from 0.3 (baseline model, Plant Village images only) to 0.5 (EfficientNetB0, with oversampled wild images). Addition of more wild images to the train dataset is expected to further increase the accuracy. Another approach to improve accuracy would be to introduce a leaf segmentation step, either as a product feature or as a preprocessing step before training.
## Setup
Recommended python version: 3.11.3.

macOS / Linux:
```
# Optional: if using pyenv
pyenv local 3.11.3

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Windows PowerShell
```
# Optional: if using pyenv
pyenv local 3.11.3

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Windows Git Bash
```
# Optional: if using pyenv
pyenv local 3.11.3

python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
