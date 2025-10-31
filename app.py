from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os
import joblib
import pickle

app = Flask(__name__)

MODEL_FILE = 'earthquake_magnitude_rfr.pkl'       # your trained model (optional)
COLUMNS_FILE = 'model_columns.pkl'                # optional saved feature column order
FALLBACK_COLUMNS = ['Depth_km','Foreshock_Count','Aftershock_Count','Energy_Released_Joules']


def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            print('Loaded model with joblib from', MODEL_FILE)
            return model
        except Exception as e:
            try:
                with open(MODEL_FILE,'rb') as f:
                    model = pickle.load(f)
                print('Loaded model with pickle from', MODEL_FILE)
                return model
            except Exception as e2:
                print('Failed to load model:', e, e2)
    else:
        print(f'Model file {MODEL_FILE} not found. Prediction will not work until you place it in the folder.')
    return None

# Utility: load columns if available
def load_columns():
    if os.path.exists(COLUMNS_FILE):
        try:
            cols = joblib.load(COLUMNS_FILE)
            print('Loaded columns from', COLUMNS_FILE)
            return list(cols)
        except Exception as e:
            try:
                with open(COLUMNS_FILE,'rb') as f:
                    cols = pickle.load(f)
                print('Loaded columns (pickle) from', COLUMNS_FILE)
                return list(cols)
            except Exception as e2:
                print('Failed to load columns:', e, e2)
    print('No columns file found — using fallback columns:', FALLBACK_COLUMNS)
    return FALLBACK_COLUMNS

MODEL = load_model()
MODEL_COLUMNS = load_columns()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Read numeric inputs from form
        try:
            depth = float(request.form.get('Depth_km', 0))
        except:
            depth = 0.0
        try:
            foreshock = int(request.form.get('Foreshock_Count', 0))
        except:
            foreshock = 0
        try:
            aftershock = int(request.form.get('Aftershock_Count', 0))
        except:
            aftershock = 0
        try:
            energy = float(request.form.get('Energy_Released_Joules', 0))
        except:
            energy = 0.0

        # Prepare input DataFrame according to MODEL_COLUMNS (or fallback)
        input_df = pd.DataFrame(columns=MODEL_COLUMNS)
        # initialize zeros
        input_df.loc[0] = 0
        # set known numeric features if present
        for col, val in [('Depth_km', depth), ('Foreshock_Count', foreshock),
                         ('Aftershock_Count', aftershock), ('Energy_Released_Joules', energy)]:
            if col in input_df.columns:
                input_df.at[0, col] = val
        prediction = None
        error = None
        if MODEL is not None:
            try:
                pred = MODEL.predict(input_df)[0]
                prediction = float(pred)
            except Exception as e:
                error = str(e)
        else:
            error = f'Model file not found. Place {MODEL_FILE} in the project folder.'

        return render_template('result.html',
                               prediction=prediction,
                               error=error,
                               inputs={
                                   'Depth_km': depth,
                                   'Foreshock_Count': foreshock,
                                   'Aftershock_Count': aftershock,
                                   'Energy_Released_Joules': energy
                               })
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
