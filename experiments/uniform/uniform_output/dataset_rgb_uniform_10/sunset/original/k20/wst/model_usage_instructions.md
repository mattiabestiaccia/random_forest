# Model Usage Instructions

## Loading the Model
```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# Load the trained model
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_selector = joblib.load('feature_selector.joblib')

# Load feature information
with open('feature_names.json', 'r') as f:
    feature_info = json.load(f)
    selected_features = feature_info['selected_features']
```

## Making Predictions
```python
# Assuming you have new_images (list of RGB images) and extract_features function
feature_data = []
for img in new_images:
    features = extract_features(img, "wst")  # You need this function
    feature_data.append(features)

X_new = np.array(feature_data)

# Apply the same preprocessing pipeline
X_new_scaled = scaler.transform(X_new)
X_new_selected = feature_selector.transform(X_new_scaled)

# Make predictions
predictions = model.predict(X_new_selected)
prediction_probabilities = model.predict_proba(X_new_selected)
```

## Model Details
- **Dataset**: original - sunset area
- **Feature Method**: Wavelet Scattering Transform
- **Selected Features**: ['R_wst_mean_4', 'R_wst_mean_11', 'R_wst_mean_13', 'R_wst_mean_34', 'R_wst_mean_52', 'R_wst_std_0', 'G_wst_mean_3', 'G_wst_mean_4', 'G_wst_mean_5', 'G_wst_mean_11', 'G_wst_mean_12', 'G_wst_mean_13', 'G_wst_mean_36', 'G_wst_mean_44', 'G_wst_mean_45', 'G_wst_mean_53', 'G_wst_mean_54', 'G_wst_std_4', 'G_wst_std_5', 'G_wst_std_12']
- **Model**: Random Forest with 50 estimators
- **Classes**: ['garden', 'low_veg', 'trees']
- **Test Accuracy**: 0.9167
- **CV Accuracy**: 0.9167 ± 0.0645
