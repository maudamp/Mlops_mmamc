import pickle
import pytest
import pandas as pd

# Charger le modèle et le scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Nouvelles données à tester
new_data = {
    'customer_id': 7442532,
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 1958.928726,
    'total_debt_outstanding': 8228.75252,
    'income': 26648.43525,
    'years_employed': 2,
    'fico_score': 572
}
def test_predict():
    # Créer un DataFrame à partir des nouvelles données
    new_data_df = pd.DataFrame([new_data])
    
    # Normaliser les nouvelles données
    new_data_normalized = scaler.transform(new_data_df)
    
    # Prédiction
    prediction = model.predict(new_data_normalized)
    prediction_proba = model.predict_proba(new_data_normalized)  # Obtenir les probabilités des classes

    # Afficher les résultats pour analyse
    print(f"Prediction: {prediction[0]}")
    print(f"Prediction Probabilities: {prediction_proba}")

    # Test si la prédiction est correcte (comparer à la première valeur)
    assert prediction[0] == 1, f"Incorrect prediction: {prediction[0]} with probabilities {prediction_proba}"