import tkinter as tk
from tkinter import messagebox
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

slf_df = pd.read_csv('C:/java/Student-Stress-Factors-main/Student-Stress-Factors-main/data/StressLevelDataset.csv')

X = slf_df.drop(columns=['stress_level'])
y = slf_df['stress_level']

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X, y)

def preprocess_user_input(input_data):
    input_data_numeric = []
    for val in input_data:
        try:
            numeric_val = float(val) 
            input_data_numeric.append(numeric_val)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter numeric values only.")
            return None
    return [input_data_numeric]

def predict_stress_level(input_data):
    knn_prediction = knn_model.predict(input_data)
    dt_prediction = dt_model.predict(input_data)
    return knn_prediction, dt_prediction

def predict_stress_gui():
    def predict():
        input_data = [entry_vars[feature].get() for feature in X.columns]

        input_data_numeric = preprocess_user_input(input_data)
        if input_data_numeric is None:
            return

        knn_pred, dt_pred = predict_stress_level(input_data_numeric)

        stress_status = "Stressed" if any(knn_pred) or any(dt_pred) else "Not Stressed"

        messagebox.showinfo("Stress Prediction", f"The person is {stress_status}")

        num_features = len(X.columns)
        num_rows = (num_features + 2) // 3 
        plt.figure(figsize=(15, 10))
        for idx, feature in enumerate(X.columns):
            plt.subplot(num_rows, 3, idx+1)
            plt.scatter(X[feature], y, label='All Data')
            plt.scatter(input_data_numeric[0][idx], 0, color='red', label='Your Data', marker='^')
            plt.xlabel(feature)
            plt.ylabel('Stress Level')
            plt.title(f'Comparison of Your Data with All Data for {feature}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    root = tk.Tk()
    root.title("Student Stress Level Predictor")

    entry_vars = {}
    for idx, feature in enumerate(X.columns):
        label = tk.Label(root, text=feature)
        label.grid(row=idx, column=0)
        entry_vars[feature] = tk.StringVar()
        entry = tk.Entry(root, textvariable=entry_vars[feature])
        entry.grid(row=idx, column=1)

    predict_button = tk.Button(root, text="Predict Stress Level", command=predict)
    predict_button.grid(row=len(X.columns), columnspan=2)

    root.mainloop()

predict_stress_gui()