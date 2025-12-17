def add_features(df):
    """
    Create binary target variable:
    0 -> Fast Delivery
    1 -> Delayed Delivery
    """

    df['delivery_status'] = df['Delivery_Time'].apply(
        lambda x: 0 if x <= 30 else 1
    )

    # Drop original delivery time column
    df.drop(columns=['Delivery_Time'], inplace=True)

    return df
