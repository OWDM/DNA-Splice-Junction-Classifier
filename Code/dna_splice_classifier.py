# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('/kaggle/input/splicejunction-gene-sequences-dataset/dna.csv')

# Data exploration and preprocessing
#print(data.head())
#print(data.describe())
#missing_values = data.isnull().sum()
#print("Missing values in each column:\n", missing_values)

# Adjust class labels to start from 0
data['class'] = data['class'] - 1

# Split data into features and target
X = data.drop('class', axis=1)
y = data['class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the stacking model
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred_stack = stacking_model.predict(X_test)
print("Accuracy with Stacking:", accuracy_score(y_test, y_pred_stack))
print("\nClassification Report with Stacking:\n", classification_report(y_test, y_pred_stack))
