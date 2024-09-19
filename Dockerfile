# Utiliser une image Python 3.11 complète
FROM python:3.11

# Installer les dépendances système nécessaires pour construire les bibliothèques Python
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libtiff5-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail pour l'application
WORKDIR /app

# Copier le fichier des dépendances dans le conteneur
COPY requirements.txt . 

# Mettre à jour pip, setuptools et wheel avant d'installer les dépendances
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Installer les dépendances Python spécifiées dans requirements.txt
RUN pip install --no-cache-dir streamlit==1.23.0
RUN pip install --no-cache-dir pandas==2.0.3
RUN pip install --no-cache-dir scikit-learn==1.5.2
RUN pip install --no-cache-dir matplotlib==3.7.1
RUN pip install --no-cache-dir seaborn==0.12.2
RUN pip install --no-cache-dir shap==0.41.0
RUN pip install --no-cache-dir plotly==5.11.0

# Installer AWS CLI pour interagir avec Amazon S3
RUN pip install awscli

# Copier le reste du code de l'application dans le conteneur
COPY . .

# Commande pour exécuter l'application
CMD ["streamlit", "run", "app.py"]
