# StudyAlgorithm
Algorithme prédictif pour optimiser les révisions

## Plan détaillé
1. Collecter des données
2. Comprendre les données (colonnes similaires, manquantes)
3. Mettre en place l’environnement de la db avec SQLite
4. Prendre un `main dataset`
5. Normaliser / Nettoyer les données pour les mettre sur les mêmes échelles
6. Créer un tableau final avec les colonnes du `main dataset` + celles ajoutées
7. Séparer 80% des données pour le `training` et 20% pour les `tests`
8. Choisir un modèle pertinent (ex : regression linéaire) pour définir des corrélations
9. Mettre en place l’environnement d’entraînement en Python
10. Entrainer le modèle → à se renseigner

### Obtenir des data sur :
- Sommeil
- Moyenne actuelle
- Heures passées à réviser 
- Nombre de sessions de révision
- Niveau de stress
- Difficulté du test
- Taux d'absence en cours
Kaggle : https://www.kaggle.com/datasets

### Insights :
- **Sommeil vs révision trade-off**
- Heures de révision minimales pour passer
- Détecter les sessions de révision inutiles