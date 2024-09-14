import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset
data = pd.read_csv('leukemia_microarray_data.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Apply ADASYN for balancing the dataset
adasyn = ADASYN()
X_res, y_res = adasyn.fit_resample(X, y)

# Feature selection using Chi-squared
chi2_selector = SelectKBest(chi2, k=100)  # Select top 100 features
X_kbest = chi2_selector.fit_transform(X_res, y_res)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y_res, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the base models
log_reg = LogisticRegression()
svc = SVC(probability=True)
extra_trees = ExtraTreesClassifier()

# Create the hybrid model using VotingClassifier
hybrid_model = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('svc', svc),
    ('et', extra_trees)
], voting='soft')

# Train the model
hybrid_model.fit(X_train, y_train)

# Evaluate the model
scores = cross_val_score(hybrid_model, X_test, y_test, cv=5)
print(f'Accuracy: {np.mean(scores) * 100:.2f}%')
