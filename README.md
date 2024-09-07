pip install --upgrade jinja2
pip install setuptools
pip install nbformat>=4.2.0
pip install scikit-learn
pip install tensorflow

# Importations de bibliothèques standard et scientifiques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px

# Importations de Scikit-Learn (Machine Learning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Importations de MLflow (suivi des expériences)
import mlflow
from mlflow.tracking import MlflowClient

# Utilitaires divers
from pprint import pprint

# Importations de bibliothèques standard et scientifiques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px

# Importations de Scikit-Learn (Machine Learning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Importations de MLflow (suivi des expériences)
import mlflow
from mlflow.tracking import MlflowClient

# Utilitaires divers
from pprint import pprint

# Lire le fichier CSV
df = pd.read_csv('Loan_Data.csv')

# Afficher les premières lignes du dataframe
df.head(4)

#Colonnes à arrondir et nettoyer
columns_to_round = ['loan_amt_outstanding', 'total_debt_outstanding', 'income']

# Arrondir les valeurs et convertir en entiers
for col in columns_to_round:
    df[col] = df[col].round().astype(int)

# Afficher les premières lignes du dataframe nettoyé pour vérification
print(df.head(2))

df.describe()

df.shape

df.dtypes

# Sélectionner les colonnes pertinentes pour détecter les outliers
cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']

# Créer des box plots pour chaque colonne
plt.figure(figsize=(15, 10))

for i, col in enumerate(cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot de {col}')

plt.tight_layout()
plt.show()

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Utilisation de la fonction
columns_to_clean = ['loan_amt_outstanding', 'total_debt_outstanding', 'income', 'fico_score']
df_cleaned = remove_outliers(df, columns_to_clean)

def detect_and_print_first_outlier(df, columns):
    for col in columns:
        # Calculer les quartiles
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Définir les bornes inférieure et supérieure
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identifier les outliers
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        
        # Afficher la première valeur des outliers si elle existe
        if not outliers.empty:
            print(f"Première valeur outlier pour '{col}': {outliers[col].iloc[0]}")
        else:
            print(f"Aucun outlier détecté pour '{col}'.")

# Liste des colonnes à analyser
cols = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']

# Appeler la fonction pour afficher les outliers
detect_and_print_first_outlier(df_cleaned, cols)

# Visualiser les box plots pour chaque colonne
plt.figure(figsize=(15, 10))
for i, col in enumerate(cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df_cleaned[col])
    plt.title(f'Box Plot de {col}')
plt.tight_layout()
plt.show()

df_cleaned.describe()

def print_first_outliers(df, columns):
    for col in columns:
        # Calculer les quartiles et l'IQR
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        # Définir les bornes inférieure et supérieure
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identifier les outliers
        outliers2 = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]

        # Afficher la première valeur des outliers si elle existe
        if not outliers2.empty:
            print(f"Première valeur outlier pour '{col}': {outliers2[col].iloc[0]}")
        else:
            print(f"Aucun outlier détecté pour '{col}'.")

# Colonnes à vérifier pour les outliers
columns_to_check = ['loan_amt_outstanding', 'total_debt_outstanding']

# Afficher les premières valeurs des outliers
print_first_outliers(df_cleaned, columns_to_check)

# Créer des box plots pour les colonnes 'loan_amt_outstanding' et 'total_debt_outstanding'
plt.figure(figsize=(12, 6))

# Box plot pour 'loan_amt_outstanding'
plt.subplot(1, 2, 1)
sns.boxplot(y=df_cleaned['loan_amt_outstanding'])
plt.title('Box Plot de loan_amt_outstanding')

# Box plot pour 'total_debt_outstanding'
plt.subplot(1, 2, 2)
sns.boxplot(y=df_cleaned['total_debt_outstanding'])
plt.title('Box Plot de total_debt_outstanding')

plt.tight_layout()
plt.show()

# Filtrer les lignes où 'loan_amt_outstanding' est supérieur à 7757
df_outliers = df_cleaned[df_cleaned['loan_amt_outstanding'] > 7757]

# Afficher les lignes où 'loan_amt_outstanding' est supérieur à 7757
print("Données où 'loan_amt_outstanding' > 7757:")

# Filtrer les lignes où 'total_debt_outstanding' est supérieur à 18858
df_outliers = df_cleaned[df_cleaned['total_debt_outstanding'] > 18858]

# Afficher les lignes où 'total_debt_outstanding' est supérieur à 18858
print("Données où 'total_debt_outstanding' > 18858:")

df_cleaned.shape

df = df_cleaned 

def analyze_data_distribution(df, columns):
    """
    Cette fonction analyse la distribution des données pour les colonnes spécifiées dans un DataFrame.
    
    Args:
    df (pd.DataFrame): Le DataFrame contenant les données.
    columns (list): La liste des colonnes à analyser.
    """
    # Définir la taille de la figure
    plt.figure(figsize=(18, 12))
    
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Pour les variables catégorielles ou de type object, afficher un diagramme en barres
            sns.countplot(data=df, x=col)
            plt.title(f'Distribution de {col}')
            plt.xticks(rotation=45)  # Rotation des labels de l'axe des x si nécessaire
        else:
            # Pour les variables numériques, afficher un histogramme
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f'Distribution de {col}')
        
        plt.xlabel(col)
        plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.show()

# Liste des colonnes à analyser
columns_to_analyze = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score', 'default']

# Analyser la distribution des données
analyze_data_distribution(df, columns_to_analyze)

def analyze_column_distribution(df, column):
    """
    Analyse la distribution d'une colonne spécifique dans un DataFrame.
    
    Args:
    df (pd.DataFrame): Le DataFrame contenant les données.
    column (str): Le nom de la colonne à analyser.
    """
    plt.figure(figsize=(10, 6))
    
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        # Pour les variables catégorielles ou de type object, afficher un diagramme en barres
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution de {column}')
        plt.xticks(rotation=45)  # Rotation des labels de l'axe des x si nécessaire
    else:
        # Pour les variables numériques, afficher un histogramme avec une courbe de densité
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Distribution de {column}')
        plt.xlabel(column)
        plt.ylabel('Fréquence')
        
    plt.tight_layout()
    plt.show()
    
# Analyser les valeurs uniques et les statistiques descriptives
print("Valeurs uniques pour 'years_employed':")
print(df['years_employed'].unique())
print("\nStatistiques descriptives pour 'years_employed':")
print(df['years_employed'].describe())

# Analyser la distribution de 'years_employed'
analyze_column_distribution(df, 'years_employed')

# Afficher les valeurs manquantes pour chaque colonne
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne:")
print(missing_values)

# Définir les intervalles pour chaque colonne
def categorize_column(df, column, bins, labels):
    """
    Convertit une colonne numérique en colonne catégorielle en fonction des intervalles spécifiés.
    
    Args:
    df (pd.DataFrame): Le DataFrame contenant les données.
    column (str): Le nom de la colonne à transformer.
    bins (list): Les bornes des intervalles.
    labels (list): Les labels pour les catégories.
    
    Returns:
    pd.DataFrame: Le DataFrame avec la colonne catégorielle ajoutée.
    """
    df[column + '_category'] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    return df

# Définir les intervalles et labels pour chaque colonne
bins_dict = {
    'credit_lines_outstanding': [0, 2, 4, 5],
    'loan_amt_outstanding': [0, 1000, 3000, 5000, 7000, 8000],
    'total_debt_outstanding': [0, 2000, 4000, 6000, 8000, 10000, 12000, 15000, 20000],
    'income': [0, 20000, 40000, 60000, 80000, 100000, 120000],
    'years_employed': [0, 1, 3, 5, 10]
}

labels_dict = {
    'credit_lines_outstanding': ['0-2', '2-4', '4+'],
    'loan_amt_outstanding': ['0-1K', '1K-3K', '3K-5K', '5K-7K', '7K-8K'],
    'total_debt_outstanding': ['0-2K', '2K-4K', '4K-6K', '6K-8K', '8K-10K', '10K-12K', '12K-15K', '15K-20K'],
    'income': ['0-20K', '20K-40K', '40K-60K', '60K-80K', '80K-100K', '100K-120K'],
    'years_employed': ['0-1Y', '1Y-3Y', '3Y-5Y', '5Y-10Y']
}

# Appliquer la transformation à chaque colonne
for column in bins_dict.keys():
    df = categorize_column(df, column, bins_dict[column], labels_dict[column])

# Afficher les premières lignes pour vérifier les résultats
print(df.head())

# Afficher la distribution des nouvelles colonnes catégorielles
for column in bins_dict.keys():
    print(f'\nDistribution de {column}_category:')
    print(df[column + '_category'].value_counts())

df.columns

# Fonction pour créer un graphique en camembert avec Plotly
def plot_pie_chart(df, column):
    """
    Crée un graphique en camembert pour une colonne catégorielle en affichant les pourcentages.
    
    Args:
    df (pd.DataFrame): Le DataFrame contenant les données.
    column (str): Le nom de la colonne catégorielle à visualiser.
    """
    # Calculer les fréquences et pourcentages
    counts = df[column].value_counts()
    percentages = (counts / counts.sum() * 100).reset_index()
    percentages.columns = [column, 'Percentage']
    
    # Créer le graphique en camembert
    fig = px.pie(percentages, names=column, values='Percentage', title=f'Distribution de {column}', 
                 labels={column: column, 'Percentage': 'Pourcentage'})
    
    # Mettre à jour le layout pour une meilleure visualisation
    fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Afficher le graphique
    fig.show()

# Liste des colonnes catégorielles à analyser
categorical_columns = [
    'credit_lines_outstanding_category',
    'loan_amt_outstanding_category',
    'total_debt_outstanding_category',
    'income_category',
    'years_employed_category'
]

# Créer des graphiques en camembert pour chaque colonne catégorielle
for column in categorical_columns:
    plot_pie_chart(df, column)


# Sélectionner les variables explicatives (features) et la variable cible (target)
X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
y = df['default']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construire et entraîner le modèle
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Faire des prédictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Évaluer le modèle
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")

# Visualiser les coefficients du modèle
coefficients = pd.DataFrame(model.coef_.flatten(), X.columns, columns=['Coefficient'])
coefficients.plot(kind='bar')
plt.title('Coefficients de la Régression Logistique')
plt.show()

