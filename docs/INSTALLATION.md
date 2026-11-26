# Installation et dépannage

Ce guide explique comment installer le projet sous Windows et comment résoudre les problèmes courants rencontrés lors de `pip install`.

## 1. Prérequis

- Python 3.12 (vérifier avec `python --version`)
- Git
- Environ 1 Go d'espace libre (xgboost, scipy et pyarrow sont volumineux)

## 2. Installation

```bat
git clone https://github.com/Okba829/DM-PROJECT.git
cd DM-PROJECT
pip install -r requirements.txt
```

## 3. Lancer l'application et les tests

```bat
python -m streamlit run app/streamlit_app.py
python -m pytest tests/
```

L'application s'ouvre sur `http://localhost:8501`.

> Si la commande `streamlit` n'est pas reconnue, utiliser `python -m streamlit`
> (le dossier des scripts Python n'est pas dans le PATH).

## 4. Dépannage

### Téléchargement très lent ou `ReadTimeoutError`

La connexion est trop lente pour les gros paquets. Augmenter le délai et le nombre d'essais :

```bat
pip install --default-timeout=200 --retries 10 -r requirements.txt
```

Relancer la même commande reprend là où elle s'est arrêtée : les paquets déjà installés ou en cache sont ignorés.

### Installer les gros paquets un par un

```bat
pip install --default-timeout=200 --retries 10 scipy
pip install --default-timeout=200 --retries 10 xgboost lightgbm
pip install --default-timeout=200 --retries 10 streamlit
```

### `OSError: [Errno 28] No space left on device`

Le disque (ou le dossier temporaire) est plein.

```bat
fsutil volume diskfree C:
pip cache purge
```

Pour utiliser un autre disque comme dossier temporaire (valable pour la fenêtre en cours) :

```bat
mkdir D:\tmp
set TEMP=D:\tmp
set TMP=D:\tmp
pip install --no-cache-dir -r requirements.txt
```

### Installation hors ligne à partir de fichiers `.whl`

Télécharger les fichiers `.whl` (cp312, win_amd64) depuis pypi.org dans un dossier existant, puis :

```bat
pip install --no-index --find-links C:\wheels scipy xgboost pyarrow
```

Le dossier doit exister et contenir les fichiers, sinon pip l'ignore.

### `fatal: not a git repository`

La commande git a été lancée en dehors du dépôt. Se placer d'abord dans le dossier du projet :

```bat
cd C:\chemin\vers\DM-PROJECT
git status
```


## Lancement de l'application : problèmes fréquents

### `'streamlit' n'est pas reconnu`

Le script `streamlit` n'est pas toujours dans le `PATH` de Windows. Lancer plutôt l'application via Python
(le fichier de l'application est `app/streamlit_app.py`, et non `app.py`) :

```
python -m streamlit run app/streamlit_app.py
```

### Streamlit demande une adresse e-mail au premier lancement

Au premier lancement, Streamlit affiche `Email:` et attend une réponse : la fenêtre semble bloquée.
Appuyer sur **Entrée** sans rien saisir, ou créer une fois pour toutes le fichier de configuration
(après `Ctrl+C` pour arrêter le programme en cours) :

```
mkdir %USERPROFILE%\.streamlit
echo [general] > %USERPROFILE%\.streamlit\credentials.toml
echo email = "" >> %USERPROFILE%\.streamlit\credentials.toml
```

Relancer ensuite l'application : la question n'est plus posée. L'application est alors disponible
sur `http://localhost:8501`.

### `fatal: not a git repository`

Les commandes `git` doivent être exécutées **dans le dossier du projet** (celui qui contient `src/`,
`tests/` et le dossier caché `.git`), pas dans le dossier personnel :

```
cd <chemin>\DM-PROJECT
git status
```

### La suite de tests est très lente

Sur un ordinateur peu puissant, `python -m pytest tests/` peut durer plusieurs minutes
(environ 7 minutes mesurées), car certains tests entraînent de vrais modèles. Pour travailler
sur un seul fichier :

```
python -m pytest tests/test_evaluate.py -q
```
