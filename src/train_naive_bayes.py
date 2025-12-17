# ================================
# Naive Bayes - Food Delivery Time Prediction
# ================================

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from data_preprocessing import load_and_preprocess_data
from feature_engineering import add_features
from evaluation import evaluate_model


print("====================================")
print("Running Naive Bayes Model")
print("====================================")

# Load and preprocess dataset
df = load_and_preprocess_data("data/Food_Delivery_Time_Prediction.csv")

# Feature engineering (create target variable)
df = add_features(df)

# Split features and target
X = df.drop('delivery_status', axis=1)
y = df['delivery_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate model
evaluate_model(y_test, y_pred, "Naive Bayes")

print("====================================")
print("Naive Bayes Model Completed")
print("====================================")
