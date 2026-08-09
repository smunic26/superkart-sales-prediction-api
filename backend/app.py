import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application with a descriptive name
superkart_api = Flask("SuperKart Sales Predictor")

# Load the serialized model from disk once when the server starts
# Doing this at module level avoids reloading the model on every request, improving response time
model = joblib.load("superkart_model.joblib")

# Health-check endpoint for the root URL
# Useful for verifying the server is running when accessed via browser or curl
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Online inference endpoint: accepts a single product-store record as JSON
# The input must include all 10 features the model was trained on
@superkart_api.post('/v1/predict')
def predict_sales():
    # Parse the JSON body sent in the POST request
    data = request.get_json()

    # Reconstruct the feature dictionary from the incoming JSON
    # The keys must exactly match the column names used during model training
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Wrap the sample in a DataFrame because sklearn pipelines expect tabular input
    input_df = pd.DataFrame([sample])
    # model.predict returns a numpy array; [0] extracts the single prediction value
    prediction = model.predict(input_df)[0]

    # Return the prediction as a JSON response with 2 decimal places
    return jsonify({'Predicted_Sales': round(float(prediction), 2)})

# Batch inference endpoint: accepts a CSV file with multiple records
# Processes all rows at once and returns predictions as a JSON dictionary
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():

    # Retrieve the uploaded CSV file from the multipart form data    superkart_api.run(debug=True)

    file = request.files['file']if __name__ == '__main__':

    input_df = pd.read_csv(file)# In production (Docker), Gunicorn is used instead (see Dockerfile)

# Only run Flask's built-in dev server when executing this script directly

    # Predict for all rows and convert to a plain Python list

    predictions = model.predict(input_df).tolist()    return jsonify(result)

    # Return a dictionary mapping row index (as string) to predicted sales value
    result = {str(i): round(float(p), 2) for i, p in enumerate(predictions)}
