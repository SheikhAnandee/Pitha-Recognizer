# Pitha-Recognizer
An image classification model from data collection, cleaning, model training, deployment and API integration. <br/>
Can classify 20 different types of pitha<br/>
Types of Pitha following:<br/>
1. Bhapa Pitha
2. Chitoi Pitha
3. Tel er Pitha
4. Nakshi Pitha
5. Bibikhana Pitha
6. Puli Pitha
7. Patisapta Pitha
8. Choi Pitha
9. Khejur Pitha
10. Dudh Chitoi Pitha
11. Bini Pitha
12. Pata Pitha
13. Jhinuk Pitha
14. Mera Pitha
15. Chita Pitha
16. Dudh Puli Pitha
17. Fuljhuri Pitha
18. Semai Pitha
19. Dim Shundori Pitha
20. Atikka Pitha

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
<img src = "deployment/gradio_app.png" width="500" height="500">

# API integration with GitHub Pages
The deployed model API is integrated [here](https://sheikhanandee.github.io/Pitha-Recognizer/) in GitHub Pages Website. Implementation and other details can be found in `docs` folder.