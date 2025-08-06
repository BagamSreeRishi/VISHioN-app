import os
import librosa
import numpy as np
# other imports...

def load_data():
    # your existing code to load data and extract features
    # returns X (features), y (labels)
    pass

if __name__ == "__main__":
    print("Starting data load...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples with labels: {set(y)}")

    # Add this block below your data loading
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, "baby_cry_classifier.joblib")
    joblib.dump(le, "label_encoder.joblib")
    print("Model and label encoder saved successfully.")
