# src/modeling.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def split_and_save_data(df):
    """
    Split the input DataFrame into training and testing sets, and save them to separate CSV files.

    Parameters:
    - df: The input DataFrame containing the data.

    Returns:
    - X_train: The features of the training set.
    - X_test: The features of the testing set.
    - y_train: The labels of the training set.
    - y_test: The labels of the testing set.
    """
    # Split the data
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Create directories for saving train and test data if they don't exist
    # os.makedirs('data/processed/train/', exist_ok=True)
    # os.makedirs('data/processed/test/', exist_ok=True)

    os.makedirs('../data/processed/train/', exist_ok=True)
    os.makedirs('../data/processed/test/', exist_ok=True)

    # Concatenate features and labels for train and test sets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save train and test data to separate CSV files
    train_data.to_csv('../data/processed/train/train_data.csv', index=False)
    test_data.to_csv('../data/processed/test/test_data.csv', index=False)

    # Return the split sets
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Apply MinMaxScaler to the training set and transform the test set.

    Parameters:
    - X_train: Training set features.
    - X_test: Test set features.

    Returns:
    - X_train_scaled: Scaled training set features.
    - X_test_scaled: Scaled test set features.
    """
    scaler = MinMaxScaler()

    # Fit and transform the training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test set using the same scaler
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def encode_labels(y_train, y_test):
    """
    Encode labels using LabelEncoder and keep track of label names.

    Parameters:
    - y_train: Labels in the training set.
    - y_test: Labels in the testing set.

    Returns:
    - y_train_encoded: Encoded labels in the training set.
    - y_test_encoded: Encoded labels in the testing set.
    - label_mapping: Dictionary mapping label names to their corresponding numeric values.
    """
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode labels in the training set
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Encode labels in the testing set
    y_test_encoded = label_encoder.transform(y_test)

    # Create a dictionary mapping label names to numeric values
    label_mapping = {label: encoded_value for label, encoded_value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

    # Assuming y_train and y_test are the original label arrays
    # Assuming label_mapping, y_train_encoded, and y_test_encoded are already computed

    # Printing the top 5 instances in the training set
    print("Top 15 instances in the training set:")
    for original_label, encoded_label in zip(y_train[:15], y_train_encoded[:15]):
        print(f"Original Label: {original_label}, Encoded Label: {encoded_label}")

    # Printing the top 5 instances in the testing set
    print("\nTop 15 instances in the testing set:")
    for original_label, encoded_label in zip(y_test[:15], y_test_encoded[:15]):
        print(f"Original Label: {original_label}, Encoded Label: {encoded_label}")

    # Printing the label mapping
    print("\nLabel Mapping:")
    for label, encoded_value in label_mapping.items():
        print(f"Original Label: {label}, Encoded Label: {encoded_value}")

    return y_train_encoded, y_test_encoded, label_mapping

def apply_pca(X_train, X_test):
    """
    Apply PCA to the scaled features and plot explained variance ratio.

    Parameters:
    - X_train: Scaled training set features.
    - X_test: Scaled test set features.

    Returns:
    - X_train_pca: Transformed training set features after PCA.
    - X_test_pca: Transformed test set features after PCA.
    """
    pca = PCA()

    # Fit and transform the training set
    X_train_pca = pca.fit_transform(X_train)

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained Variance Ratio - Cumulative')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # Save the variance ratio as an image file
    plt.savefig('./output/images/explained_variance.png')

    # Add a comment indicating that the countplot image has been saved
    print("Explained variance ratio plot saved as 'explained_variance.png'")

    # Determine the optimal number of components (capturing 95% of the variance)
    n_components_pca = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_.cumsum() < 0.95])
    print("The number of principal components explaining 95% of information:", n_components_pca)

    # Apply PCA with the selected number of components
    pca = PCA(n_components=n_components_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca

def balance_data_with_smote_and_undersample(X_train, y_train, sampling_strategy='auto'):
    """
    Balance the dataset by using SMOTE for oversampling and RandomUnderSampler for undersampling.

    Parameters:
    - X_train: Training set features.
    - y_train: Training set labels.
    - sampling_strategy: Strategy to balance the dataset. 'auto' means equalizing the number of samples in each class.

    Returns:
    - X_train_balanced: Balanced training set features.
    - y_train_balanced: Balanced training set labels.
    """
    # Create a pipeline for combining SMOTE and RandomUnderSampler
    pipeline = make_pipeline(SMOTE(sampling_strategy=sampling_strategy), RandomUnderSampler(sampling_strategy=sampling_strategy))

    # Fit and transform the training set using the pipeline
    X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced

def train_classifier(model, X_train, y_train):
    """
    Train a specified classifier on the scaled and balanced data.

    Parameters:
    - model: The classifier model to be trained.
    - X_train: Scaled and balanced training set features.
    - y_train: Scaled and balanced training set labels.

    Returns:
    - model: Trained classifier model.
    """
    print("Training started for " + str(model))
    model.fit(X_train, y_train)
    print("\nTraining completed!")

    return model

def get_key_by_value(dictionary, target_value):
    """
    Retrieve the key from a dictionary based on a given value.

    Parameters:
    - dictionary: The dictionary to search.
    - target_value: The value to find in the dictionary.

    Returns:
    - key: The key corresponding to the target value. If not found, returns None.
    """
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the value is not found, you may choose to return a default value or raise an exception.
    return None  # or raise ValueError("Value not found in the dictionary")

def evaluate_model_and_save(model, X_test, y_test, label_mapping, classifier_name="classifier", output_folder="../output/evaluation_results"):
    """
    Evaluate a classification model using various metrics, visualize the confusion matrix,
    and save all output to a file.

    Parameters:
    - model: The trained classifier.
    - X_test: Test set features.
    - y_test: True labels for the test set.
    - label_mapping: Dictionary mapping label names to their corresponding numeric values.
    - classifier_name: Name of the classifier for filename.
    - output_folder: Folder path to save the output files.

    Returns:
    - accuracy: Accuracy of the model.
    - classification_rep: Classification report.
    """
    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, zero_division=1)

    # Generate a timestamp for filename uniqueness
    timestamp = datetime.now().strftime("%d%m%Y%H%M%S")

    # Formulate the output file path with the classifier name
    output_file_path = f"{output_folder}/{classifier_name}_evaluation_{timestamp}.txt"

    # Open the file for writing
    with open(output_file_path, "w") as output_file:
        # Write evaluation metrics to the file
        output_file.write(f"\nAccuracy: {accuracy:.2f}\n")
        output_file.write("Classification Report:\n")
        output_file.write(f"{classification_rep}\n")
        output_file.write(f"The Accuracy of the Model is {accuracy_score(y_test, y_pred)}\n")
        output_file.write(f"The Precision of the Model is {f1_score(y_test, y_pred, average='weighted')}\n")
        output_file.write(f"The Recall of the Model is {precision_score(y_test, y_pred, average='weighted')}\n")
        output_file.write(f"The F1 Score of the Model is {recall_score(y_test, y_pred, average='weighted')}\n")

        # Visualize the confusion matrix and save the plot
        plt.figure(figsize=(15, 15))
        confusion_matrix_data = confusion_matrix(y_test, y_pred)
        sns.heatmap(confusion_matrix_data, annot=True)
        plt.savefig(f"{output_folder}/{classifier_name}_confusion_matrix_{timestamp}.png")
        plt.close()

        # Write the confusion matrix path to the file
        output_file.write(f"\nConfusion Matrix saved as {classifier_name}_confusion_matrix_{timestamp}.png\n")
        
        # Print the top 100 instances with actual and predicted labels using label names
        output_file.write("\nTop 50 instances with actual and predicted labels:\n")
        for i in range(50):
            actual_label_name = get_key_by_value(label_mapping, y_test[i])
            predicted_label_name = get_key_by_value(label_mapping, y_pred[i])
            output_file.write(f"Instance {i + 1}: Actual Label - {actual_label_name}, Predicted Label - {predicted_label_name}\n")
    print(f"Output saved to {output_file_path}")

    return accuracy, classification_rep