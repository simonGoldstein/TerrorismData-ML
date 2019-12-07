ML Final Project
----
By Erin Clark, Mason English, Simon Goldstein, Austin Promenschenkel

### About
This is a machine learning project to both learn about machine learning through predicting information about terrorism attacks.
We got our data from Kaggle. The Cookbook is the codebook from our dataset. The data is a link to kaggle. And my github copy is a link to this github. 

Link to [cookbook](https://start.umd.edu/gtd/downloads/Codebook.pdf)                                        
                                                                                                            
Link to our [data](https://www.kaggle.com/START-UMD/gtd)                                                    
                                                                                                            
My github copy of the [this project](https://github.com/simonGoldstein/TerrorismData-ML/blob/master/gtd.zip)

### How to Run

Download the repo

    git clone https://github.com/simonGoldstein/TerrorismData-ML.git

Install Libraries
    
    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install scikit-learn==0.21.3
    pip install dash==1.7.0
    pip install dash-bootstrap-components

To unzip data and run models:
    
    cd data
    unzip gtd.zip
    cd ../modelCode
    python3 model_to_run.py

To fetch trained models and run UI:

    chmod +x fetchTrainedModels.sh
    ./fetchTrainedModels.sh
    python3 app.py
