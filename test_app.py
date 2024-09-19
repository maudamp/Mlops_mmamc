import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Charger les données (remplacez par votre propre fichier CSV)
df = pd.read_csv('Loan_Data.csv')

# Diviser en caractéristiques (features) et cible (target)
X = df.drop(columns=["default"])  # Exclure la colonne cible 'default'
y = df["default"]  # La colonne cible 'default'

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Normaliser les données d'entraînement
X_val = scaler.transform(X_val)  # Normaliser les données de validation en utilisant le même scaler

# Initialiser et entraîner le modèle
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle et le scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)  # Sauvegarder le modèle dans 'model.pkl'

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)  # Sauvegarder le scaler dans 'scaler.pkl'

print("Modèle et scaler sauvegardés dans model.pkl et scaler.pkl")
