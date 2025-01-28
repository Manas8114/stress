import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the existing dataset from the CSV file
slf_df = pd.read_csv('C:/java/Student-Stress-Factors-main/Student-Stress-Factors-main/data/StressLevelDataset.csv')

# Define the features (X) and target (y) from the existing dataset
X = slf_df.drop(columns=['stress_level'])
y = slf_df['stress_level']

# Train the KNN and Decision Tree models
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X, y)

# Function to preprocess user input and convert to numeric values
def preprocess_user_input(input_data):
    input_data_numeric = []
    for val in input_data:
        try:
            numeric_val = float(val)  # Convert input to numeric value
            input_data_numeric.append(numeric_val)
        except ValueError:
            # Handle non-numeric values (e.g., display an error message)
            messagebox.showerror("Input Error", "Please enter numeric values only.")
            return None
    return [input_data_numeric]

# Function to predict stress level based on user input
def predict_stress_level(input_data):
    knn_prediction = knn_model.predict(input_data)
    dt_prediction = dt_model.predict(input_data)
    return knn_prediction, dt_prediction

# GUI setup
def predict_stress_gui():
    def predict():
        # Get user input from GUI fields
        input_data = [entry_vars[feature].get() for feature in X.columns]

        # Preprocess the input data to numeric values
        input_data_numeric = preprocess_user_input(input_data)
        if input_data_numeric is None:
            return

        # Predict stress level using models
        knn_pred, dt_pred = predict_stress_level(input_data_numeric)

        # Determine if the person is stressed or not
        stress_status = "Stressed" if any(knn_pred) or any(dt_pred) else "Not Stressed"

        # Display the prediction to the user
        messagebox.showinfo("Stress Prediction", f"The person is {stress_status}")

        # Create plots
        create_plots(input_data_numeric[0])

    def create_plots(input_data_numeric):
        # Compare input data with all data and create a scatter plot for all features
        num_features = len(X.columns)
        num_rows = (num_features + 2) // 3  # Calculate the number of rows needed for subplots
        fig, axs = plt.subplots(num_rows, 3, figsize=(15, 10))

        for idx, feature in enumerate(X.columns):
            row = idx // 3
            col = idx % 3
            axs[row, col].scatter(X[feature], y, label='All Data')
            axs[row, col].scatter(input_data_numeric[idx], 0, color='red', label='Your Data', marker='^')
            axs[row, col].set_xlabel(feature)
            axs[row, col].set_ylabel('Stress Level')
            axs[row, col].set_title(f'Comparison of Your Data with All Data for {feature}')
            axs[row, col].legend()

        # Print correlation for stress_level
        corr = slf_df.corr()
        print(np.abs(corr['stress_level']).sort_values(ascending=False))

        # Plot correlation matrix
        sns.heatmap(corr, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axs[-1, -1])
        axs[-1, -1].set_title('Correlation Matrix')

        plt.tight_layout()

        # Embed the plot into tkinter window
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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

    frame = tk.Frame(root)
    frame.grid(row=len(X.columns)+1, columnspan=2)

    root.mainloop()

# Run the GUI for stress level prediction
predict_stress_gui()
