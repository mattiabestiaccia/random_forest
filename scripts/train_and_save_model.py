#!/usr/bin/env python3
"""
Standalone script to reproduce the rgb_salt_pepper25_kbest experiment
for advanced_stats_original_k5_popolar and save the trained model.

This script replicates the exact experiment configuration and saves the final trained model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from scipy.ndimage import sobel, laplace
import warnings
warnings.filterwarnings('ignore')

# Configuration matching the original experiment
CONFIG = {
    'dataset_path': '/home/brusc/Projects/random_forest/dataset_rgb_salt_and_pepper_25/original',
    'area_name': 'popolar',
    'feature_method': 'advanced_stats',
    'k_features': 5,
    'output_dir': '/home/brusc/Projects/random_forest/experiments/rgb_salt_pepper25_kbest/experiments/advanced_stats_original_k5_popolar',
    'n_estimators': 50,
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

def load_rgb_image(file_path):
    """Load RGB PNG image using PIL"""
    image = Image.open(file_path).convert('RGB')
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return image_array

def extract_advanced_features(rgb_image):
    """Extract advanced statistical features from each RGB channel"""
    features_per_channel = 18
    features = np.zeros(3 * features_per_channel)
    
    for i in range(3):
        channel = rgb_image[i]
        ch_flat = channel.ravel()
        ch_clean = ch_flat[np.isfinite(ch_flat)]
        
        if len(ch_clean) == 0:
            continue
            
        base = i * features_per_channel
        
        # Basic statistics
        features[base + 0] = np.mean(ch_clean)
        features[base + 1] = np.std(ch_clean)
        features[base + 2] = np.var(ch_clean)
        features[base + 3] = np.min(ch_clean)
        features[base + 4] = np.max(ch_clean)
        features[base + 5] = np.ptp(ch_clean)  # range
        
        # Shape statistics
        features[base + 6] = stats.skew(ch_clean)
        features[base + 7] = stats.kurtosis(ch_clean)
        mean_val = features[base + 0]
        features[base + 8] = features[base + 1] / max(mean_val, 1e-8)  # coefficient of variation
        
        # Percentiles
        features[base + 9] = np.percentile(ch_clean, 10)
        features[base + 10] = np.percentile(ch_clean, 25)
        features[base + 11] = np.percentile(ch_clean, 50)
        features[base + 12] = np.percentile(ch_clean, 75)
        features[base + 13] = np.percentile(ch_clean, 90)
        features[base + 14] = features[base + 12] - features[base + 10]  # IQR
        
        # MAD
        features[base + 15] = np.mean(np.abs(ch_clean - mean_val))
        
        # Gradient and edge density
        try:
            grad_x = sobel(channel, axis=0)
            grad_y = sobel(channel, axis=1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            features[base + 16] = np.mean(grad_mag.ravel())
            
            edges = np.abs(laplace(channel))
            edge_thr = np.percentile(edges.ravel(), 90)
            features[base + 17] = np.mean(edges.ravel() > edge_thr)
        except:
            features[base + 16] = 0
            features[base + 17] = 0
    
    return features

def load_area_images_and_labels(root_dir, area_name):
    """Load RGB images and their labels from a specific area"""
    images = []
    labels = []
    
    area_path = os.path.join(root_dir, area_name)
    
    if not os.path.exists(area_path):
        raise ValueError(f"Area directory not found: {area_path}")
    
    class_dirs = [d for d in os.listdir(area_path) if os.path.isdir(os.path.join(area_path, d))]
    class_dirs.sort()
    
    print(f"Found classes in {area_name}: {class_dirs}")
    
    for class_dir in class_dirs:
        class_path = os.path.join(area_path, class_dir)
        png_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
        
        print(f"Loading {len(png_files)} images from {area_name}/{class_dir}")
        
        for png_file in tqdm(png_files, desc=f"Loading {area_name}/{class_dir}"):
            file_path = os.path.join(class_path, png_file)
            try:
                image = load_rgb_image(file_path)
                images.append(image)
                labels.append(class_dir)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    return images, np.array(labels)

def select_features_kbest(X, y, feature_names, k=5):
    """Apply SelectKBest with Mutual Information"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    feature_scores = selector.scores_[selected_indices]
    
    return X_selected, selected_features, feature_scores, scaler, selector

def train_final_model(X, y, test_size=0.2, random_state=42):
    """Train the final model on full dataset with train/test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=CONFIG['n_estimators'],
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation on full dataset
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    
    return rf, {
        'test_accuracy': test_accuracy,
        'cv_mean_accuracy': float(np.mean(cv_scores)),
        'cv_std_accuracy': float(np.std(cv_scores)),
        'cv_scores': cv_scores.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def save_model_and_artifacts(model, scaler, selector, selected_features, feature_scores, 
                           performance_results, dataset_info, output_dir):
    """Save the trained model and all artifacts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the Random Forest model
    model_path = os.path.join(output_dir, 'trained_model.joblib')
    joblib.dump(model, model_path)
    print(f"Saved trained model: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")
    
    # Save the feature selector
    selector_path = os.path.join(output_dir, 'feature_selector.joblib')
    joblib.dump(selector, selector_path)
    print(f"Saved feature selector: {selector_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'feature_scores': feature_scores.tolist() if isinstance(feature_scores, np.ndarray) else feature_scores,
            'all_feature_names': get_feature_names()
        }, f, indent=2)
    print(f"Saved feature information: {features_path}")
    
    # Save comprehensive experiment report
    report = {
        "experiment_name": "KBest_Advanced_Stats_RGB_Original_Popolar_k5_WithModel",
        "config": CONFIG,
        "dataset_info": dataset_info,
        "feature_selection": {
            "method": f"SelectKBest_k{CONFIG['k_features']}",
            "num_features": len(selected_features),
            "selected_features": selected_features,
            "feature_scores": feature_scores.tolist() if isinstance(feature_scores, np.ndarray) else feature_scores
        },
        "performance": performance_results,
        "model_files": {
            "trained_model": "trained_model.joblib",
            "scaler": "scaler.joblib",
            "feature_selector": "feature_selector.joblib",
            "feature_names": "feature_names.json"
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    report_path = os.path.join(output_dir, 'experiment_report_with_model.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved experiment report: {report_path}")
    
    # Save model usage instructions
    usage_instructions = '''# Model Usage Instructions

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
# Assuming you have new_images (list of RGB images) and extract_advanced_features function
feature_data = []
for img in new_images:
    features = extract_advanced_features(img)  # You need this function
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
- **Dataset**: RGB Salt & Pepper 25 - Original - Popolar area
- **Feature Method**: Advanced Statistics (54 features -> 5 selected)
- **Selected Features**: ''' + str(selected_features) + '''
- **Model**: Random Forest with ''' + str(CONFIG['n_estimators']) + ''' estimators
- **Classes**: ''' + str(list(dataset_info['classes'].keys())) + '''
- **Test Accuracy**: ''' + f"{performance_results['test_accuracy']:.4f}" + '''
- **CV Accuracy**: ''' + f"{performance_results['cv_mean_accuracy']:.4f} ± {performance_results['cv_std_accuracy']:.4f}" + '''
'''
    
    usage_path = os.path.join(output_dir, 'model_usage_instructions.md')
    with open(usage_path, 'w') as f:
        f.write(usage_instructions)
    print(f"Saved usage instructions: {usage_path}")

def get_feature_names():
    """Get the feature names for advanced statistics"""
    stat_names = [
        'mean', 'std', 'var', 'min', 'max', 'range', 'skew', 'kurt', 'cv',
        'p10', 'p25', 'p50', 'p75', 'p90', 'iqr', 'mad', 'grad_mean', 'edge_density'
    ]
    return [f"{c}_{stat}" for c in ['R', 'G', 'B'] for stat in stat_names]

def main():
    print("="*80)
    print("STANDALONE MODEL TRAINING SCRIPT")
    print("Reproducing: rgb_salt_pepper25_kbest/advanced_stats_original_k5_popolar")
    print("="*80)
    
    # Load images and labels
    print(f"Loading images from {CONFIG['area_name']} area...")
    images, labels = load_area_images_and_labels(CONFIG['dataset_path'], CONFIG['area_name'])
    
    if len(images) == 0:
        raise ValueError(f"No images found for area: {CONFIG['area_name']}")
    
    print(f"Loaded {len(images)} images with {len(np.unique(labels))} classes")
    print(f"Classes: {list(np.unique(labels))}")
    print(f"Image shape: {images[0].shape}")
    
    # Extract features
    print("Extracting advanced statistical features...")
    feature_data = []
    feature_names = get_feature_names()
    
    for img in tqdm(images, desc="Extracting features"):
        features = extract_advanced_features(img)
        feature_data.append(features)
    
    X = np.array(feature_data)
    print(f"Feature matrix shape: {X.shape}")
    
    # Dataset info
    class_counts = pd.Series(labels).value_counts()
    dataset_info = {
        "data_directory": CONFIG['dataset_path'],
        "area_name": CONFIG['area_name'],
        "total_images": len(images),
        "classes": class_counts.to_dict(),
        "image_shape": list(images[0].shape),
        "total_features_available": len(feature_names),
        "feature_method": CONFIG['feature_method'],
        "dataset_type": "rgb_original_salt_pepper25",
        "k_features": CONFIG['k_features']
    }
    
    # Feature selection
    print(f"Applying SelectKBest feature selection (k={CONFIG['k_features']})...")
    X_selected, selected_features, feature_scores, scaler, selector = select_features_kbest(
        X, labels, feature_names, k=CONFIG['k_features']
    )
    
    print(f"Selected features: {selected_features}")
    print(f"Feature scores: {feature_scores}")
    
    # Train final model
    print("Training final Random Forest model...")
    model, performance_results = train_final_model(
        X_selected, labels, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state']
    )
    
    # Save everything
    print("Saving model and artifacts...")
    save_model_and_artifacts(
        model, scaler, selector, selected_features, feature_scores,
        performance_results, dataset_info, CONFIG['output_dir']
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Area: {CONFIG['area_name']}")
    print(f"Dataset: {CONFIG['dataset_path']}")
    print(f"Feature method: {CONFIG['feature_method']}")
    print(f"Total images: {dataset_info['total_images']}")
    print(f"Classes: {list(dataset_info['classes'].keys())}")
    print(f"Original features: {dataset_info['total_features_available']}")
    print(f"Selected features: {CONFIG['k_features']}")
    print(f"N estimators: {CONFIG['n_estimators']}")
    print(f"\nPERFORMANCE RESULTS:")
    print(f"Test Accuracy: {performance_results['test_accuracy']:.4f}")
    print(f"CV Accuracy: {performance_results['cv_mean_accuracy']:.4f} ± {performance_results['cv_std_accuracy']:.4f}")
    print(f"\nTop {len(selected_features)} selected features:")
    for i, (feat, score) in enumerate(zip(selected_features, feature_scores)):
        print(f"  {i+1}. {feat}: {score:.4f}")
    print(f"\nModel and artifacts saved to: {CONFIG['output_dir']}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)