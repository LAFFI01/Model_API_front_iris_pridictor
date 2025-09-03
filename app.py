from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field
from typing import List,Optional,Annotated,Literal,Dict
import pickle
import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset to get target names
iris = load_iris()
target_names = iris.target_names

# Load the trained model from a pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the expected input data schema using Pydantic
class UserInput(BaseModel):
    # Sepal length in centimeters, must be between 0 and 10
    sepal_length: Annotated[float, Field(gt=0, lt=10,description="Sepal length in cm", example=5.1)]
    # Petal width in centimeters, must be between 0 and 10
    petal_width: Annotated[float, Field(gt=0, lt=10,description="Petal width in cm", example=0.2)]
    
# Define the prediction endpoint
@app.post('/predict')
def predict_types(input: UserInput):
    # Convert validated input to dictionary
    input_dict = input.model_dump()
    # Rename keys to match the model's expected feature names
    formatted_dict = {
        'sepal length (cm)': input_dict['sepal_length'],
        'petal width (cm)': input_dict['petal_width']
    }
    # Create a DataFrame from the input dictionary
    input_data = pd.DataFrame([formatted_dict])
    # Make prediction using the loaded model
    model_prediction = model.predict(input_data)[0]
    # Map the prediction to the class name
    class_name = target_names[model_prediction]
    # Return the class name as a JSON response
    return JSONResponse(status_code=200, content={"prediction": class_name})