# Big Data for Healthcare Final Project

## Brief Description
We seek to reproduce ["Predicting Heart Failure Readmission from Clinical Notes Using Deep Learning"](https://ieeexplore.ieee.org/document/8983095) . In this paper, they attempt to predict heart failure readmission or 30-day readmission using clinical notes from the MIMIC-III v1.4 database without doing any feature engineering. They do so by training a CNN model on word2vec vectorized words from the notes, and achieve an F1 score of 0.756 in the general readmission case and 0.733 for 30-day readmission. 

Traditionally, such analysis has been done by many intensively hand-crafted features. Their main contribution is in creating a model with powerful predictive capability that does not require expert crafted features. They also baseline on Random Forest, and use Chi-squared feature analysis to give some interpretation as to what their model believes words to mean.


## Installation Instructions
We used the following Python, Cuda, and Pytorch versions:

- Python version: 3.11.6
- Cuda: 11.8
- Pytorch: pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
![image](https://github.gatech.edu/storage/user/73259/files/782f920b-4c7c-483f-ac32-b97f2e19ed62)

### Packages
All necessary packages can be installed using the `requirements.txt` file.

### Data
If you have access to the MIMIC database, you can use the SQL file `BD4H_FINAL_PROJECT_HEARTFAILURE_READMISSION_MIMIC_QUERIES.sql` In Google Big Query to generate the dataset and export the final table "hf_admission_notes_with_target_labels" as a CSV file.

### Pre-Trained Model
The name of the pre-trained model we used is "PubMed-and-PMC-w2v.bin." It can be downloaded from this URL: http://evexdb.org/pmresources/vec-space-models/. Once downloaded, it should be placed in the `vector_embeddings` folder located at the root of the repository.

### How to Run
Once you have the CSV file and the pre-trained model downloaded and placed in the aforementioned folders, you should be able to run the Python notebook `BD4H_Readmission_Prediction_CNN_RF_Final.ipynb` end-to-end. Each section in this notebook is properly documented with headings and comments for ease of understanding. It includes data pre-processing, tokenizing, obtaining word embeddings, CNN model training and evaluation, Random Forest training and evaluation. Additionally, it features a chi-square analysis of key features to explain the prediction results. The only dependency is utility.py. 


### Authors
* **Pratt, Luke O (pratt.o.luke@gmail.com)** 
* **Siddiqui, Muhammad N (msiddiqui43@gatech.edu)** 


