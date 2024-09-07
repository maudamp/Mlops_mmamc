import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tempfile
import io
import plotly.graph_objects as go
import plotly.express as px

# Charger le modèle et le scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Interface Streamlit
st.title('Prédiction de Risque de Défaut de Paiement')

# Entrées dans la barre latérale pour les données utilisateur
customer_id = st.sidebar.number_input('Identifiant du Client', min_value=0, step=1, help="Entrez l'identifiant unique du client.")
credit_lines_outstanding = st.sidebar.number_input('Nombre de Lignes de Crédit en Cours', min_value=0, help="Entrez le nombre de lignes de crédit actives pour le client.")
loan_amt_outstanding = st.sidebar.number_input('Montant du Prêt en Cours (€)', min_value=0, help="Entrez le montant total du prêt en cours du client en Euro.")
total_debt_outstanding = st.sidebar.number_input('Dette Totale en Cours (€)', min_value=0, help="Entrez le montant total de la dette du client en Euro.")
income = st.sidebar.number_input('Revenus Annuels du Client (€)', min_value=0, help="Entrez le revenu annuel du client en Euro.")
years_employed = st.sidebar.number_input("Nombre d'Années d'Emploi", min_value=0, step=1, help="Entrez le nombre d'années d'emploi du client dans son poste actuel.")
fico_score = st.sidebar.number_input('Score FICO', min_value=300, max_value=850, help="Entrez le score FICO du client (entre 300 et 850).")

# Créer un DataFrame avec les données utilisateur
user_input = pd.DataFrame(
    [[customer_id, credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]],
    columns=['customer_id', 'credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
)

# Normaliser les données d'entrée
user_input_normalized = scaler.transform(user_input)

# Prédiction
prediction = model.predict(user_input_normalized)

# Afficher le résultat de la prédiction
st.subheader('Résultat de la Prédiction')
if prediction[0] == 1:
    st.write('Le client présente un risque de défaut de paiement.')
else:
    st.write('Le client ne présente pas de risque de défaut de paiement.')

# Bouton pour afficher les graphiques si les données sont complètes
if st.button('Prédir'):
    # 1. Graphique du Ratio d'Endettement
    debt_ratio = total_debt_outstanding / income if income != 0 else 0

    # Créer un graphique en barres avec Plotly pour le ratio d'endettement
    debt_ratio_fig = go.Figure()
    debt_ratio_fig.add_trace(go.Bar(x=['Ratio d\'Endettement'], y=[debt_ratio], marker_color='skyblue'))
    debt_ratio_fig.update_layout(
        title='Ratio d\'Endettement',
        yaxis_title='Ratio',
        xaxis_title='Métrique',
        xaxis=dict(tickvals=['Ratio d\'Endettement']),
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )
    st.plotly_chart(debt_ratio_fig)

    # 2. Graphique de la Distribution de la Dette Totale
    df = pd.read_csv('Loan_Data.csv')

    # Créer un histogramme avec Plotly pour la distribution de la dette totale
    debt_distribution_fig = px.histogram(df, x='total_debt_outstanding', nbins=30, title='Distribution de la Dette Totale')
    debt_distribution_fig.update_layout(
        xaxis_title='Dette Totale (€)',
        yaxis_title='Fréquence',
        template='plotly_white'
    )
    st.plotly_chart(debt_distribution_fig)

    # Si la prédiction indique un risque, afficher les variables importantes
    if prediction[0] == 1:
        # Préparer l'explicateur SHAP en utilisant le modèle et les données
        X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
        explainer = shap.Explainer(model, X)

        # Calculer les valeurs SHAP pour les données d'entrée de l'utilisateur
        user_input_shap_values = explainer(user_input_normalized)

        # Afficher les valeurs SHAP
        st.subheader('Importance des Variables')
        st.write("Les variables les plus importantes pour la prédiction sont :")

        # Sauvegarder le graphique summary de SHAP dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            shap.summary_plot(user_input_shap_values, user_input, feature_names=user_input.columns, show=False)
            plt.savefig(tmpfile.name)
            plt.close()
            st.image(tmpfile.name)

        # Afficher les valeurs SHAP pour la première instance
        st.subheader('Valeurs SHAP pour les Données Utilisateur')
        shap_values_df = pd.DataFrame(user_input_shap_values.values, columns=user_input.columns)
        st.write(shap_values_df)

