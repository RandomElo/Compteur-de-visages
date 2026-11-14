# Compteur de visages

Script Python qui compte les visages sur des photos de groupe et ajoute le résultat dans un fichier CSV. Pensé pour suivre l'effectif d'un groupe, séance après séance.

## Fonctionnement

1. Les photos sont déposées dans `entree/`, nommées `JJ-MM-AAAA.jpg` (jpg, jpeg ou png).
2. Les visages sont détectés avec [InsightFace](https://github.com/deepinsight/insightface) (modèle `buffalo_l`). Les détections qui se chevauchent (IoU > 0,5) sont fusionnées.
3. Une image annotée est enregistrée dans `imagesAnalysees/`, la photo d'origine est déplacée dans `ancienne/`.
4. Une ligne est ajoutée à `effectifs.csv` : date (`AAAA/MM/JJ`), jour de la semaine, nombre de personnes.

Extrait de `effectifs.csv` :

| date | jour | nombre_personnes |
|---|---|---|
| 2025/09/09 | mardi | 24 |

## Installation

Python 3.10 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Au premier lancement, le modèle `buffalo_l` est téléchargé automatiquement.

## Utilisation

```bash
python script.py
```

Les noms de fichiers qui ne respectent pas le format `JJ-MM-AAAA` sont ignorés avec un avertissement.

## Vie privée

Les photos contiennent des personnes. Les dossiers `entree/`, `ancienne/`, `faces/` et `imagesAnalysees/` ne doivent pas être versionnés.
