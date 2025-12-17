import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop ID column (not useful for prediction)
    df.drop(columns=['Order_ID'], inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.columns.drop('Delivery_Time')
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
