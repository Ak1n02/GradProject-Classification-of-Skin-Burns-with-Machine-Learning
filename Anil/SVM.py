import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset from the CSV file
df = pd.read_csv(r'./Anil/dataset.csv')

# Check if data is loaded properly
print(f"Dataset loaded with shape: {df.shape}")
print(df.head())

# Features (X) and Target (y)
X = df[['mean_a', 'std_a', 'skew_a', 'kurtosis_a', 'mean_b', 'std_b', 'skew_b', 'kurtosis_b', 'hue_mean', 'hue_std']]
y = df['degree']  # Degree is the target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling: SVM works better with normalized data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM model
svm_model = SVC(kernel='linear', random_state=69, C=0.1, gamma="scale")
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")

# Optional: Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(SVC(random_state=42), X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean() * 100:.4f}%")

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

