# AI Model — Plan d'entraînement sur les données `datas/`

## 0. Contexte des données

Le dossier [datas/](datas/) contient des mesures mensuelles (Mars 2018 → Octobre 2022, ~55 fichiers `.pickle`) issues du dataset *Guided Waves from Long-Term Structural Health Monitoring under Uncontrolled and Dynamic Conditions* (Figshare 28112504).

Chaque fichier `measurements YYYY_MM.pickle` est un `dict` Python (~2 Go) contenant ~4 600 mesures :

| Clé | Forme | Description |
|---|---|---|
| `datatime` | (N,) | Horodatage de chaque mesure |
| `temperature`, `pressure`, `brightness`, `humidity` | (N,) | Variables environnementales |
| `excitation signal` | (1000,) | Signal d'excitation commun à toutes les mesures (1 ms) |
| `guided wave` | (N, 8, 2000) | Réponse ultrasonore — 8 chemins (5-1..5-4, 6-1..6-4), 2000 échantillons chacun |
| `damage tag` | (N,) | **Étiquette cible** — binaire {0 = sain, 1 = endommagé} |
| `weather tag` | (N,) | Catégorie météo {0..5} (utile comme feature de robustesse) |

> ⚠️ Les descriptions dans `data inf` inversent `damage tag` et `weather tag` — utiliser **les noms de colonnes**, pas les descriptions. Classe positive ~4.5 % → **fort déséquilibre**.

Total estimé : ~250 000 mesures × 16 000 valeurs de signal = données volumineuses → impose un pipeline **streaming + features compressées**.

---

## 1. Objectif du modèle

**Tâche principale** : classification binaire `damage tag` à partir des 8 ondes guidées + features environnementales/temporelles.

**Tâches secondaires** (extension naturelle vers le pitch "Building Health Profile") :
- Score continu de "santé structurelle" (probabilité calibrée + tendance lissée mensuelle).
- Détection d'anomalie non supervisée (autoencodeur sur signaux sains uniquement) → utile car la classe positive est rare et son étiquetage peut être imparfait.

---

## 2. Architecture du pipeline

```
datas/*.pickle
   │
   ├── 1. Streaming loader (un mois à la fois, jamais tout en RAM)
   │
   ├── 2. Préprocessing du signal
   │      • normalisation par canal
   │      • alignement / fenêtrage autour de l'écho
   │      • extraction de features (FFT, ondelettes, énergie par bande)
   │
   ├── 3. Stockage intermédiaire compact (Parquet, ~50× plus petit)
   │
   ├── 4. Split temporel (pas aléatoire !)
   │      • train : 2018-03 → 2020-12
   │      • val   : 2021-01 → 2021-06
   │      • test  : 2021-07 → 2022-10
   │
   ├── 5. Modèle
   │
   └── 6. Évaluation + calibration + export
```

---

## 3. Étapes détaillées

### Étape 1 — Mise en place de l'environnement
- Python 3.11, env virtuel dédié.
- Dépendances : `numpy`, `pandas`, `pyarrow`, `scipy`, `scikit-learn`, `torch`, `pywavelets`, `tqdm`, `matplotlib`, `mlflow` (suivi d'expériences).
- GPU recommandé (CUDA) pour la phase 2 — un CPU multi-cœur reste viable pour les baselines.

### Étape 2 — Streaming loader
- Itérateur `month_iterator()` qui ouvre **un seul** `.pickle` à la fois, yield des mini-batchs `(N_chunk, …)`.
- Aucun fichier ouvert simultanément → RAM bornée < 4 Go.

### Étape 3 — Préprocessing & feature engineering
Pour chaque mesure (8×2000 floats) :
1. Normalisation par chemin (mean/std calculés sur le **train uniquement**, persistés).
2. **Features classiques temps-fréquence** par canal :
   - Énergie totale, RMS, kurtosis, crest factor.
   - 5 bandes FFT (énergie cumulée).
   - 4 niveaux d'ondelettes (`db4`) — énergie par niveau.
   - Temps d'arrivée du premier écho (cross-correlation avec `excitation signal`).
3. Concaténation avec features environnementales (`temperature`, `pressure`, `humidity`, `brightness`) + encodage cyclique du `datatime` (heure, jour de l'année) + `weather tag` one-hot.

Résultat : ~150 features tabulaires par mesure. Sauvegarde en Parquet partitionné par mois.

### Étape 4 — Split temporel & balancing
- **Pas de split aléatoire** : le but est de prédire l'évolution future → split chronologique strict.
- Gérer le déséquilibre (~4.5 % positifs) :
  - `class_weight='balanced'` pour les baselines.
  - Pour le deep learning : focal loss ou oversampling raisonné (jamais sur le val/test).

### Étape 5 — Modélisation (3 niveaux croissants)

**Niveau A — Baseline tabulaire** (1 jour de travail)
- **XGBoost** ou **LightGBM** sur les ~150 features.
- Objectif : poser une borne basse honnête, identifier les features les plus importantes, valider que le déséquilibre est gérable.
- Métriques cibles : ROC-AUC > 0.85, PR-AUC > 0.30.

**Niveau B — Modèle profond 1D-CNN sur signaux bruts** (3-5 jours)
- Entrée : `(8, 2000)` ondes guidées normalisées.
- Architecture : 4 blocs `Conv1D + BatchNorm + ReLU + MaxPool` → GAP → MLP → sigmoid.
- Concaténation tardive des features environnementales avant le MLP final.
- Loss : `BCEWithLogits` + focal (γ=2).
- Cible : ROC-AUC > 0.92 sur la fenêtre de test 2021-2022.

**Niveau C — Détection d'anomalies non supervisée** (extension, 2-3 jours)
- Autoencodeur convolutionnel entraîné **uniquement sur les mesures saines** de 2018-2019.
- Score d'anomalie = erreur de reconstruction.
- Permet de couvrir les types de dommages absents du train et alimente le "Health Score" du produit.

### Étape 6 — Évaluation
- **Matrice de confusion + courbes ROC/PR** sur l'ensemble de test (2021-07 → 2022-10).
- **Dérive temporelle** : courbe du F1 mois par mois → vérifier que le modèle ne se dégrade pas avec le vieillissement de la structure.
- **Robustesse météo** : performances stratifiées par `weather tag` (la valeur produit dépend du fait que l'IA ne se déclenche pas sur la pluie / vent).
- **Calibration** : courbe de fiabilité + Brier score → indispensable pour un "health score" exploitable.
- Suivi expériences via MLflow (hyperparams, métriques, artefacts).

### Étape 7 — Packaging
- Export du meilleur modèle en TorchScript (ou ONNX).
- Script `predict.py` qui prend un `.pickle` mensuel et produit un CSV `(timestamp, damage_proba, health_score)`.
- Tests d'inférence sur un mois non vu (2022-10) → mesure de la latence par mesure.

---

## 4. Structure de code proposée

```
AI_model/
├── requirements.txt
├── config.yaml              # chemins, hyperparams, splits
├── src/
│   ├── data/
│   │   ├── loader.py        # streaming pickle → batches
│   │   └── preprocessing.py # signal → features
│   ├── features/
│   │   └── build_features.py  # script principal datas/*.pickle → parquet/
│   ├── models/
│   │   ├── baseline_gbm.py
│   │   ├── cnn1d.py
│   │   └── autoencoder.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_inspection.ipynb
│   └── 03_results_analysis.ipynb
└── artifacts/               # checkpoints, scalers, modèles exportés
```

> Une base existe déjà dans [AI_1/DATASET/bridge_shm/](AI_1/DATASET/bridge_shm/) — à inspecter et réutiliser si compatible plutôt que repartir de zéro.

---

## 5. Risques & mitigations

| Risque | Mitigation |
|---|---|
| Taille des données (~100 Go décompressé) | Streaming + Parquet, jamais de chargement complet. |
| Déséquilibre extrême | Focal loss, métriques PR-AUC plutôt que accuracy. |
| Étiquettes potentiellement bruitées (description du dataset ambigüe) | Croiser avec l'autoencodeur non supervisé. |
| Fuite temporelle | Split chronologique strict, scalers fittés sur train uniquement. |
| Dérive saisonnière (temp/humidité) | Inclure `weather tag` et features cycliques, évaluer par strate. |

---

## 6. Jalons proposés

| Semaine | Livrable |
|---|---|
| S1 | Loader + EDA + features tabulaires sur 3 mois |
| S2 | Baseline GBM entraîné sur tout le dataset, premier rapport de métriques |
| S3-4 | CNN 1D + tuning, comparaison vs baseline |
| S5 | Autoencodeur + health score continu |
| S6 | Calibration, packaging, démo `predict.py` sur un mois inédit |
