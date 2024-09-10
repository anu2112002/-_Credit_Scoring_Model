import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Sample dataset
data = pd.DataFrame({
    'Credit_Score': [700, 650, 720, 680, 690],
    'Income': [50000, 45000, 60000, 48000, 55000],
    'Loan_Amount': [15000, 20000, 10000, 25000, 12000],
    'Loan_Term': [36, 48, 24, 60, 36],
    'Employment_Status': ['Employed', 'Self-Employed', 'Employed', 'Unemployed', 'Employed'],
    'Debt_to_Income_Ratio': [0.3, 0.4, 0.2, 0.5, 0.25],
    'Number_of_Dependents': [2, 1, 0, 3, 1],
    'Account_Balance': [2000, 1500, 3000, 1000, 2500],
    'Previous_Delinquencies': [1, 2, 0, 3, 1],
    'Age': [35, 40, 30, 45, 33],
    'Education_Level': ["Bachelor's", "Master's", "Bachelor's", "High School", "Bachelor's"],
    'Marital_Status': ['Married', 'Single', 'Married', 'Single', 'Married'],
    'Credit_Worthiness': [1, 0, 1, 0, 1]
})

# Preprocess data
# Convert categorical variables to numerical format
data = pd.get_dummies(data, columns=['Employment_Status', 'Education_Level', 'Marital_Status'])

# Split data into features and target variable
X = data.drop('Credit_Worthiness', axis=1)
y = data['Credit_Worthiness']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print Accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Print Confusion Matrix
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

# Print Classification Report with zero_division parameter
print(f'Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}')
