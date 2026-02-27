from flask import Flask, render_template, request, jsonify
import pickle, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Chargement des modèles ───────────────────────────────────────────────────
def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

census_model    = load_pkl('models/census.pkl')
census_features = load_pkl('models/census_features.pkl')
census_scores   = load_pkl('models/census_scores.pkl')
auto_model      = load_pkl('models/auto-mpg.pkl')
auto_scaler     = load_pkl('models/auto_scaler.pkl')
auto_scores     = load_pkl('models/auto_scores.pkl')

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ─── PARTIE 1 : Classification Census ─────────────────────────────────────────
@app.route('/census', methods=['GET', 'POST'])
def census():
    prediction = None
    proba      = None
    error      = None

    if request.method == 'POST':
        try:
            row = {
                'age':              float(request.form['age']),
                'education-num':    float(request.form['education_num']),
                'hours-per-week':   float(request.form['hours_per_week']),
                'capital-gain':     float(request.form['capital_gain']),
                'capital-loss':     float(request.form['capital_loss']),
                'education':        request.form['education'],
                'marital-status':   request.form['marital_status'],
                'occupation':       request.form['occupation'],
                'sex':              request.form['sex'],
            }
            df_input = pd.DataFrame([row])
            df_encoded = pd.get_dummies(df_input, drop_first=True)

            # Aligner avec les features d'entraînement
            for col in census_features:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[census_features]

            pred = census_model.predict(df_encoded)[0]
            if hasattr(census_model, 'predict_proba'):
                proba = round(census_model.predict_proba(df_encoded)[0][pred] * 100, 1)
            prediction = "Revenu > 50K" if pred == 0 else "Revenu ≤ 50K"
        except Exception as e:
            error = str(e)

    return render_template('census.html',
                           prediction=prediction, proba=proba,
                           error=error, scores=census_scores)

# ─── PARTIE 2 : Régression Auto-MPG ───────────────────────────────────────────
@app.route('/autompg', methods=['GET', 'POST'])
def autompg():
    prediction = None
    error      = None

    if request.method == 'POST':
        try:
            features = np.array([[
                float(request.form['cylinders']),
                float(request.form['displacement']),
                float(request.form['horsepower']),
                float(request.form['weight']),
                float(request.form['acceleration']),
                float(request.form['model_year']),
                float(request.form['origin']),
            ]])
            if auto_scaler:
                features = auto_scaler.transform(features)
            pred = auto_model.predict(features)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            error = str(e)

    return render_template('autompg.html',
                           prediction=prediction, error=error,
                           scores=auto_scores)

# ─── API JSON ─────────────────────────────────────────────────────────────────
@app.route('/api/predict/census', methods=['POST'])
def api_census():
    try:
        data = request.get_json()
        df_input   = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df_input, drop_first=True)
        for col in census_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[census_features]
        pred  = int(census_model.predict(df_encoded)[0])
        label = "Revenu > 50K" if pred == 0 else "Revenu ≤ 50K"
        return jsonify({'prediction': pred, 'label': label, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/api/predict/autompg', methods=['POST'])
def api_autompg():
    try:
        data     = request.get_json()
        features = np.array([[data['cylinders'], data['displacement'],
                               data['horsepower'], data['weight'],
                               data['acceleration'], data['model_year'], data['origin']]])
        if auto_scaler:
            features = auto_scaler.transform(features)
        pred = float(auto_model.predict(features)[0])
        return jsonify({'prediction': round(pred, 2), 'unit': 'mpg', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
