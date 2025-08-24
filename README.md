# Modeles-ML-interpretables
ML appliqué à la détection de fraude bancaire, avec interprétabilité (SHAP) 

Cette solution permet de :

*Charger un dataset de transactions bancaires avec une classe de catégorsation (fraude : 1, non fraude : 0), ou laisser le dataset par défaut "creditcard.csv" de https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
*Choisir un modèle de classification parmi : RF, XGBoost, LightGBM, Logistic Regression
*Choisir Une technique de rééchantillonage si le dataset est déséquilibré : Undersampling (NearMiss, RandomUnderSampler), Oversampling (SMOTE, ADASYN, BorderlineSMOTE)
*Pamarétrer le modèle choisi
*Afficher les résultats :
  - prédection du modèle choisi : Chiffres (nombre de fraudes détectées, nombre de transactions normales détectées ..)
  - Métriques (accuracy, recall, precision, etc)
  - Graphe de l'importance des variables
  - Distribution des variables par classe
  - Graphes SHAP (summary plot, waterfall)

