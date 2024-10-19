import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

############################
# Load the dataset
data = pd.read_csv('health_practices.csv')

# Prepare the features and labels (add new columns: sleep, diet, stress)
X = data[['water_intake', 'exercise', 'working_hours', 'sleep', 'diet', 'stress']]
y = data['tips']

# Encode the tips to numerical format
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and the label encoder for later use
joblib.dump(model, 'health_practices_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

##############################
