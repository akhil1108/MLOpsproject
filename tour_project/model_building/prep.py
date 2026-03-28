# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = f"hf://datasets/Akhilesh1108/Project/tourism.csv"
tourist_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the Classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                       # Age of Customer
    'CityTier',                  #City Tier
    'DurationOfPitch',           # Time taken to complete a pitch
    'NumberOfPersonVisiting',    # People visited during the pitch
    'NumberOfFollowups',         # Total number of follow ups done post pitch
    'PreferredPropertyStar',     # Prefered property
    'NumberOfTrips',             # number of trips customer takes anually
    'PitchSatisfactionScore',     # Score indicating the customer's satisfaction
    'MonthlyIncome',              # Gross monthly income of the customer
    'OwnCar',                     # Whether the customer owns a car
    'Passport',                    #does the customer holds a passport
    'NumberOfChildrenVisiting',   #number of children below 5 yrs
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # How customer contacted
    'Occupation',            # Occupation of customer
    'Gender',                 # Gender of customer
    'MaritalStatus',          #marital status of cutomer
    'Designation',            # work designation
    'ProductPitched',         # product pitched
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourist_dataset[numeric_features + categorical_features]

# Define target variable
y = tourist_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.4,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Akhilesh1108/Project",
        repo_type="dataset",
    )
