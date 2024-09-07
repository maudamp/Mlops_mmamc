# Utiliser une image Python complète plutôt que slim pour inclure plus de bibliothèques par défaut
FROM python:3.12

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

# Copier uniquement le fichier des dépendances pour optimiser le cache
COPY requirements.txt .

# Installer les dépendances Python spécifiées dans requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copier le reste du code de l'application dans le conteneur
COPY . .

# Commande pour exécuter l'application
CMD ["python", "app.py"]