import pandas as pd
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load processed data
# ==========================================

print("\nLoading processed dataset...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 2. Create models
# ==========================================

models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        kernel="rbf",
        C=10,
        probability=True
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 3. Create models folder
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 4. Train and evaluate
# ==========================================

results = {}

print("\n==========================================")
print("       TRAINING ML MODELS")
print("==========================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # Confusion matrix
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # Save model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )

    print(f"Saved: models/{filename}.pkl")


# ==========================================
# 5. Compare models
# ==========================================

print("\n==========================================")
print("          MODEL COMPARISON")
print("==========================================")

for name, accuracy in sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{name:<20} : {accuracy * 100:.2f}%"
    )


# ==========================================
# 6. Find best model
# ==========================================

best_model_name = max(
    results,
    key=results.get
)

best_accuracy = results[best_model_name]

print("\n==========================================")
print("             BEST MODEL")
print("==========================================")

print(f"Model   : {best_model_name}")
print(f"Accuracy: {best_accuracy * 100:.2f}%")

print("\nTraining completed successfully! 🎉")