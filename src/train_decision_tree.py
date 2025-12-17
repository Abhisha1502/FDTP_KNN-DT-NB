from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import load_and_preprocess_data
from feature_engineering import add_features
from evaluation import evaluate_model

print("Running Decision Tree Model...")

df = load_and_preprocess_data("data/Food_Delivery_Time_Prediction.csv")
df = add_features(df)

X = df.drop('delivery_status', axis=1)
y = df['delivery_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

evaluate_model(y_test, y_pred, "Decision Tree")

print("Decision Tree completed.")
