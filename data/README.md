# Données

Ce dossier doit contenir les fichiers **NSL-KDD** (non versionnés car volumineux, voir `.gitignore`) :

| Fichier | Contenu |
|---|---|
| `KDDTrain+.txt` | jeu d'entraînement (environ 125 000 enregistrements) |
| `KDDTest+.txt` | jeu de test (environ 22 500 enregistrements, contient des attaques absentes du train) |

Source : page NSL-KDD du *Canadian Institute for Cybersecurity* (Université du Nouveau-Brunswick). Les fichiers `.txt` sont des CSV sans en-tête : 41 attributs, l'étiquette, puis un niveau de difficulté. Le code (`src/config.py`) fournit les noms de colonnes.

Pour étendre le projet : UNSW-NB15 et CIC-IDS2017 (même démarche, adapter `config.py` et `data_loader.py`).
