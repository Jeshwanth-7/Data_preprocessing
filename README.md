# Data_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset creation
data = {
    'Age': [25, 32, 47, 51, 62],
    'Salary': [50000, 64000, 120000, 110000, 150000],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original DataFrame:\n", df)
print("\nScaled Training Features:\n", X_train_scaled)
print("\nScaled Testing Features:\n", X_test_scaled)
print("sample code for git")
