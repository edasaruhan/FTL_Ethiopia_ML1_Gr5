# ml_backend/ml_model/train_model.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import json
# Load data
df = pd.read_csv("C:/Users/Liya/Desktop/FLP/FTL_Ethiopia_ML1_Gr5/backend/ml_backend/core/ml_model/Synthetic_Student_Learning_DataM.csv") 
# Drop unnecessary columns and clean
columns_to_drop = ['Consent \n', 'Timestamp', 'Email Address', 'personalized recommendations?']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()]
str_cols = df.select_dtypes(include=['object']).columns
df[str_cols] = df[str_cols].astype(str).apply(lambda x: x.str.strip())

# Encoding
entmnt_mapper = {'Less than 1 hr': 0.5, '1 - 2 hrs': 1.5, '2 - 4 hrs': 3.0, '> 4 hrs': 4.5}
internet_mapper = {'No': 1, 'Sometimes': 2, 'Yes': 3}

if 'Avg hrs on Digtal Entmnt' in df.columns:
    df['Avg hrs on Digtal Entmnt_Encoded'] = df['Avg hrs on Digtal Entmnt'].map(entmnt_mapper)
if 'Internet access' in df.columns:
    df['Internet access_Encoded'] = df['Internet access'].map(internet_mapper)
if 'Age' in df.columns:
    df['Age'] = df['Age'].astype(str).str.extract(r'(\d+)').astype(float)

if 'Avg hrs on online class' in df.columns:
    df['Online Class Hrs Encoded'] = df['Avg hrs on online class'].map(entmnt_mapper)
    df['Total Hrs Online'] = df['Avg hrs on Digtal Entmnt_Encoded'] + df['Online Class Hrs Encoded']

df.drop(['Avg hrs on Digtal Entmnt', 'Internet access', 'Avg hrs on online class'], axis=1, errors='ignore', inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Define features and target
target_col = 'Materials used'
leakage_cols = ['XBS', 'TB', 'Exam Questions', 'Ethio Matric', 'YouTube: OCT',
                'YouTube: Blackpen Redpen', 'Royal Math', 'Galaxy Math',
                'YouTube: The Secret', 'YouTube: Ethioeducation',
                'Top Exam Book', 'Key Exam Book', 'Indian Websites (Byjus)',
                'Telegram Channels (Notes and Questions)']
drop_cols = leakage_cols + ['Timestamp', 'Email Address']
drop_cols += [col for col in df.columns if 'Consent' in col]

X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
y = df[target_col]

# Remove potential leakage from other sources
leak_columns = [col for col in X.columns if 'score' in col.lower() or 'result' in col.lower() or 'grade' in col.lower()]
X.drop(columns=leak_columns, errors='ignore', inplace=True)

# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Build model pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Save the best model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
joblib.dump(grid_search.best_estimator_, MODEL_PATH)

# Save feature names used during training
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'model_feature_names.json')
with open(FEATURES_PATH, 'w') as f:
    json.dump(X_train.columns.tolist(), f)


    

print("✅ Model training complete and saved to:", MODEL_PATH)
print("📊 Final Training Features:")
for col in X_train.columns:
    print(f"- {col}: {X_train[col].dtype}")
