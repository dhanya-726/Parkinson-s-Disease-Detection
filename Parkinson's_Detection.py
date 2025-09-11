# Parkinson's Disease Detection

# ✅ Importing the necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
warnings.filterwarnings('ignore')
sns.set()

# ✅ Install opendatasets and download the dataset
# pip install opendatasets --quiet
import opendatasets

username = input("Enter your Kaggle username: ")
key = input("Enter your Kaggle API key: ")
{"username": username, "key": key}

dataset_url = 'https://www.kaggle.com/datasets/thecansin/parkinsons-data-set?datasetId=409297&sortBy=voteCount'
opendatasets.download(dataset_url)

# ✅ Read data
data_path = './parkinsons-data-set/parkinsons.data'
df = pd.read_csv(data_path)

# ✅ Drop name column
df = df.drop('name', axis=1)

# ✅ Visualize target distribution
df['status'].value_counts().plot(kind='bar')
plt.title('Distribution of Target variable')
plt.show()

df['status'].value_counts().plot(kind='pie', autopct='%.f%%')
plt.title('Target variable distribution')
plt.show()

# ✅ Plot function for EDA
def plots(plot_kind, dataframe):
    plot_kind = plot_kind.lower()
    plot_func = {
        'violin': sns.violinplot,
        'box': sns.boxplot,
        'histogram': sns.histplot,
        'kde': sns.kdeplot
    }

    figure = plt.figure(figsize=(25, 16))
    for index, column in enumerate(dataframe.columns):
        axis = figure.add_subplot(5, 5, index + 1)
        if plot_kind in ['violin', 'box']:
            plot_func[plot_kind](y=dataframe[column], ax=axis)
        else:
            plot_func[plot_kind](dataframe[column], ax=axis)
        axis.set_title(f'{plot_kind.capitalize()} plot for {column}')
    plt.tight_layout()
    plt.show()

# ✅ EDA Plots
plots('violin', df)
plots('box', df)
plots('histogram', df)
plots('kde', df)

# ✅ Features and target separation
features = df.drop('status', axis=1)
target = df['status']

# ✅ Train-Test Split
train_data, test_data, train_labels, test_labels = train_test_split(features, target, stratify=target, test_size=0.2, random_state=2)

# ✅ Data Normalization
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)

# ✅ Optimized Feature Selection for Random Forest
from sklearn.feature_selection import SelectKBest, f_classif

feature_range = range(5, len(features.columns) + 1, 2)
rf_scores = []
best_k = 5
best_score = 0

print("🔍 Testing different numbers of features for Random Forest optimization:")

for k in feature_range:
    selector = SelectKBest(score_func=f_classif, k=k)
    train_selected = selector.fit_transform(train_data_scaled, train_labels)
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores = cross_val_score(rf_temp, train_selected, train_labels, cv=5, scoring='accuracy')
    avg_score = cv_scores.mean()
    
    rf_scores.append(avg_score)
    print(f"Features: {k:2d} | RF CV Accuracy: {avg_score*100:.2f}")
    
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print(f"\n🎯 Optimal number of features for Random Forest: {best_k}")

plt.figure(figsize=(12, 6))
plt.plot(feature_range, rf_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Number of Features')
plt.ylabel('Random Forest CV Accuracy')
plt.title('Feature Selection Optimization for Random Forest')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Apply optimal feature selection
selector_optimal = SelectKBest(score_func=f_classif, k=best_k)
train_data_optimal = selector_optimal.fit_transform(train_data_scaled, train_labels)
test_data_optimal = selector_optimal.transform(test_data_scaled)

feature_names = features.columns.tolist()
selected_features = selector_optimal.get_support()
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
print(f"\n📋 Selected features ({len(selected_feature_names)}): {selected_feature_names}")

# ✅ Tune RandomForestClassifier with GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\n🔧 Performing hyperparameter tuning for Random Forest...")
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(train_data_optimal, train_labels)

best_rf_model = grid.best_estimator_
print(f"🎯 Best Random Forest parameters: {grid.best_params_}")

log_prediction = best_rf_model.predict(test_data_optimal)
log_accuracy = accuracy_score(log_prediction, test_labels)
print(f'🔍 Accuracy of Optimized RandomForest: {round(log_accuracy*100, 2)}%')

# ✅ Confusion Matrix
log_confusion = confusion_matrix(test_labels, log_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=log_confusion)
disp.plot()
plt.title("Confusion Matrix - Optimized RandomForest")
plt.show()

# ✅ Testing multiple models with optimal feature selection
models = [
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    XGBClassifier()
]

model_results = []

def best_model(model_list):
    for model in model_list:
        model.fit(train_data_optimal, train_labels)
        prediction = model.predict(test_data_optimal)
        accuracy = accuracy_score(prediction, test_labels)
        model_results.append({
            'Model Name': type(model).__name__,
            'Model Accuracy Score': round(accuracy * 100, 2)
        })
    return pd.DataFrame(model_results).sort_values(
        by='Model Accuracy Score', ascending=False)

# ✅ Show results
results_df = best_model(models)
print("\nModel Accuracy Comparison (with optimized features):")
print(results_df)

# ✅ Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(13), x='Importance', y='Feature')
plt.title('Top 13 Feature Importance - Optimized Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ✅ Prediction function for new data
def predict_parkinsons(input_features):
    new_data = np.array([input_features])
    new_data_scaled = scaler.transform(new_data)
    new_data_selected = selector_optimal.transform(new_data_scaled)
    prediction = best_rf_model.predict(new_data_selected)[0]
    probability = best_rf_model.predict_proba(new_data_selected)[0]
    result = "Has Parkinson's" if prediction == 1 else "Healthy"
    return result

# ✅ User input prediction
user_input = input("Enter 22 comma-separated voice features: \n")
input_features = list(map(float, user_input.split(',')))
result = predict_parkinsons(input_features)
print(f"Result: {result}")

"""**Model Result comparison**

🔍 Accuracy of Optimized RandomForest: 94.87%

Model Accuracy Comparison (with optimized features):
             Model Name  Model Accuracy Score
   KNeighborsClassifier                 97.44
 RandomForestClassifier                 97.44
          XGBClassifier                 94.87
                    SVC                 82.05
"""
