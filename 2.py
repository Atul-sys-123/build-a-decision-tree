
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


file_path = "bank.csv"

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Please check the path.")
else:
    data = pd.read_csv(file_path, sep=';')  

    # Preview data
    print("Data Preview:\n", data.head())

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data.drop("y", axis=1)
    y = data["y"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
    plt.title("Decision Tree")
    plt.show()
    plt.savefig("decision_tree.png")
    
