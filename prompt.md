### **Analisi comparativa metodiche di features extraction e robustezza al rumore gaussiano**

---

### **Obiettivo**

Analizzare e confrontare le prestazioni di due approcci di estrazione delle feature — **Wavelet Scattering Transform (WST)** e **statistiche sui canali RGB** — nella classificazione di immagini affette da rumore gaussiano, utilizzando un **modello Random Forest**.

---

### **Descrizione degli esperimenti**

I dati si trovano organizzati nelle seguenti directory:

* `/home/brusc/Projects/random_forest/experiments_organized/rgb_clean_kbest`
* `/home/brusc/Projects/random_forest/experiments_organized/rgb_gaussian30_kbest`
* `/home/brusc/Projects/random_forest/experiments_organized/rgb_gaussian50_kbest`

Ogni esperimento riguarda una **combinazione** di:

* **Metodo di estrazione delle feature**: `wavelet_scattering` oppure `advanced_stats`
* **Dataset**: in tre versioni (`mini`, `small`, `original`)
* **Area geografica**: `sunset`, `assatigue`, `popolar`
* **Intensità del rumore**: pulito (`clean`), rumore gaussiano con σ = 30 (`gaussian30`), σ = 50 (`gaussian50`)

I risultati sono salvati come file JSON con la seguente struttura (esempio):

```json
{
  "experiment_name": "KBest_Advanced_Stats_RGB_Mini_Assatigue_k2",
  "area_name": "assatigue",
  "dataset_type": "mini",
  "feature_method": "advanced_stats",
  "k_features": 2,
  "dataset_info": {
    "total_images": 15,
    "classes": {
      "low_veg": 5,
      "trees": 5,
      "water": 5
    },
    "image_shape": [3, 128, 128],
    "total_features_available": 54
  },
  "feature_selection": {
    "method": "SelectKBest_k2",
    "selected_features": ["R_min", "G_max"],
    "feature_scores": [1.40, 1.01]
  },
  "performance": {
    "mean_accuracy": 0.933,
    "std_accuracy": 0.133,
    "cv_scores": [1.0, 0.667, 1.0, 1.0, 1.0],
    "n_estimators": 5
  }
}
```

---

### **Richieste**

#### 📄 1. Report comparativo

Per ciascuna configurazione sperimentale:

* Riassumere i **parametri chiave dell’esperimento** (nome esperimento, area, dataset, metodo, `k` usato, feature selezionate)
* Mostrare le **metriche di classificazione** (accuracy media, deviazione standard, cross-validation scores)
* Rilevare eventuali anomalie o variazioni elevate tra i fold

#### 📊 2. Grafici comparativi

* Accuracy e F1-score per ogni esperimento, raggruppati per area/dataset/intensità rumore
* Se possibile, curve ROC/AUC
* Tabelle riassuntive aggregate

#### 🧠 3. Analisi qualitativa

* Commentare la **robustezza della WST rispetto alle statistiche RGB**, in particolare sotto corruzione da rumore
* Valutare l’efficacia delle feature selezionate nei diversi scenari
* Identificare eventuali tendenze legate alla dimensione del dataset o all’area geografica

---

### 📌 Output attesi

* Un report finale (in formato testo, Markdown o PDF)
* Grafici salvati per ciascuna combinazione esperimento (PNG, PDF, SVG)
* Una tabella CSV o Excel con tutte le metriche aggregate
