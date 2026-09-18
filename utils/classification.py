"""
classification.py
------------------
Trains a Random Forest classifier to predict the incident category
(Database, Network, API, Server, Application) from the numeric metrics.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.preprocessing import NUMERIC_COLUMNS


def train_classifier(df):
    """
    Train a Random Forest to predict incident category from numeric metrics.

    Returns:
        model            - the trained classifier
        label_encoder    - encodes/decodes category names <-> numbers
        metrics          - dict of accuracy/precision/recall/f1 on the test set
    """
    X = df[NUMERIC_COLUMNS].values
    y_raw = df["category"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return model, label_encoder, metrics


def predict_category(model, label_encoder, feature_row):
    """
    Predict the category for a single incident.
    feature_row must be a list/array in NUMERIC_COLUMNS order.

    Returns (predicted_category, confidence, probability_dict)
    """
    probabilities = model.predict_proba([feature_row])[0]
    predicted_index = probabilities.argmax()
    predicted_category = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probabilities[predicted_index]

    probability_dict = {
        label_encoder.inverse_transform([i])[0]: prob
        for i, prob in enumerate(probabilities)
    }

    return predicted_category, confidence, probability_dict
