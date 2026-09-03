# Pitha-Recognizer
An image classification project covering the full pipeline — data collection, cleaning, model training, deployment, and API integration. <br/>

[![Status](https://img.shields.io/badge/Status-Live-brightgreen)](https://sheikhanandee.github.io/Pitha-Recognizer/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SheikhAnandee/pitha-recognizer)
[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-orange)](https://sheikhanandee.github.io/Pitha-Recognizer/)
![Repo Size](https://img.shields.io/github/repo-size/SheikhAnandee/Pitha-Recognizer)
![License](https://img.shields.io/github/license/SheikhAnandee/Pitha-Recognizer)

<!--
![Contributors](https://img.shields.io/github/contributors/SheikhAnandee/Pitha-Recognizer?color=red)
![Issues](https://img.shields.io/github/issues/SheikhAnandee/Pitha-Recognizer)
![Good First Issues](https://img.shields.io/github/issues/SheikhAnandee/Pitha-Recognizer/good%20first%20issue)
![Last Commit](https://img.shields.io/github/last-commit/SheikhAnandee/Pitha-Recognizer) 
-->
![Repo Size](https://img.shields.io/github/repo-size/SheikhAnandee/Pitha-Recognizer)
![License](https://img.shields.io/github/license/SheikhAnandee/Pitha-Recognizer)
# Overview
Pitha-Recognizer can classify **20 different types of Pitha**(traditional Bengali rice cakes) from an image, and is deployed as a live, interactive web app.

# Dataset Preparation
**Data Collection:** Downloaded from DuckDuckGo using term name <br/>
**DataLoader:** Used fastai DataBlock API to set up the DataLoader. <br/>
**Data Augmentation:** fastai provides default data augmentation which operates in GPU. <br/>
Details can be found in `notebooks/pitha_images_prep.ipynb`

# Training and Data Cleaning
**Training:** Fine-tuned a resnet34 model for 5 epochs (3 times) and got upto 90% accuracy. <br/>
**Data Cleaning:** This part took the highest time. Since I collected data from browser, there were many noises. Also, there were images that contained. I cleaned and updated data using fastai ImageClassifierCleaner. I cleaned the data each time after training or finetuning, except for the last time which was the final iteration of the model. <br/>

# Model Deployment
I deployed to model to HuggingFace Spaces Gradio App. The implementation can be found in `deployment` folder or [here](https://huggingface.co/spaces/SheikhAnandee/pitha-recognizer). <br/>
<img src = "deployment/gradio_app.png" width="1000" height="1000">

# API integration with GitHub Pages
The deployed model API is integrated [here](https://sheikhanandee.github.io/Pitha-Recognizer/) in GitHub Pages Website. Implementation and other details can be found in `docs` folder.
