import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

def visualize_label_distribution(df):
    """
    Generate a countplot to visualize the distribution of labels in the dataframe.

    Parameters:
    - df: The dataframe containing label information.
    """
    # Generate a countplot for label distribution (excluding 'BENIGN')
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y=df['label'].loc[df['label'] != 'BENIGN'])
    plt.xscale('log')  # Scale x-axis logarithmically for better visualization
    plt.title('Label Distribution (excluding BENIGN)')
    plt.xlabel('Count')
    plt.ylabel('Label')
    plt.show()

    # Save the countplot as an image file
    plt.savefig('label_distribution.png')

    # Add a comment indicating that the countplot image has been saved
    print("Label distribution countplot saved as 'label_distribution.png'")

def multiclass_pca_projection(df):
    """
    Perform PCA on the subsampled dataset and visualize the multiclass 2D PCA projection.

    Parameters:
    - df: The dataframe containing the dataset.

    Note: Subsamples the dataset (10% sample from each class) for efficient visualization.
    """
    # Subsample the dataset for efficient EDA (10% sample from each class)
    subsample_df = df.groupby('label').apply(pd.DataFrame.sample, frac=0.1).reset_index(drop=True)

    # Extract features and labels
    X = subsample_df.drop(['label'], axis=1)
    y = subsample_df['label']

    # Apply PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(X)

    # Create a DataFrame for the 2D PCA representation
    pca_df = pd.DataFrame()
    pca_df['label'] = y
    pca_df['PCA 1'] = z[:, 0]
    pca_df['PCA 2'] = z[:, 1]

    # Visualize the 2D PCA representation with a scatter plot (multiclass)
    sns.scatterplot(data=pca_df, x='PCA 1', y='PCA 2', hue='label', palette=sns.color_palette('hls', len(pca_df.label.value_counts()))).set_title("Attack Vector PCA Projection")

    # Add legend and display the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # Save the multiclass PCA plot as an image file
    plt.savefig('multiclass_pca_projection.png')

    # Add a comment indicating that the multiclass PCA plot image has been saved
    print("Multiclass PCA projection saved as 'multiclass_pca_projection.png'")

def binary_class_pca_projection(df):
    """
    Perform PCA on the dataset and visualize the binary class 2D PCA projection.

    Parameters:
    - df: The dataframe containing the dataset.

    Note: Assigns 'ATTACK' label to non-'BENIGN' labels for binary class projection.
    """
    # Binary class projection: Assign 'ATTACK' to non-'BENIGN' labels
    df.loc[df.label != 'BENIGN', 'label'] = 'ATTACK'

    # Apply PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(df.drop(['label'], axis=1))

    # Create a DataFrame for the 2D PCA representation
    pca_df = pd.DataFrame()
    pca_df['label'] = df['label']
    pca_df['PCA 1'] = z[:, 0]
    pca_df['PCA 2'] = z[:, 1]

    # Visualize the binary class 2D PCA representation with a scatter plot
    sns.scatterplot(data=pca_df, x='PCA 1', y='PCA 2', hue='label', palette=sns.color_palette('hls', 2)).set_title("Attack Vector Binary Class PCA Projection")

    # Add legend and display the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # Save the binary class PCA plot as an image file
    plt.savefig('binary_class_pca_projection.png')

    # Add a comment indicating that the binary class PCA plot image has been saved
    print("Binary class PCA projection saved as 'binary_class_pca_projection.png'")
