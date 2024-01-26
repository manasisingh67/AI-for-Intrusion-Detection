from src.data_loading import load_data
from src.data_preprocessing import data_preprocessing
from src.data_visualization import visualize_label_distribution
from src.modeling import (
    split_and_save_data,
    scale_data,
    encode_labels,
    apply_pca,
    train_classifier,
    evaluate_model_and_save
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def main(classifier_name):
    # Load and preprocess data
    df = load_data()
    df = data_preprocessing(df)
    print(df.head())

    # Visualize label distribution
    visualize_label_distribution(df)

    # Split and save data
    X_train, X_test, y_train, y_test = split_and_save_data(df)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Encode labels
    y_train_encoded, y_test_encoded, label_mapping = encode_labels(y_train, y_test)

    # Apply PCA
    X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled)

    # Train classifier
    if classifier_name.lower() == "bernoullinb":
        model = train_classifier(BernoulliNB(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "randomforest":
        model = train_classifier(RandomForestClassifier(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "knn":
        model = train_classifier(KNeighborsClassifier(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "histgradientboosting":
        model = train_classifier(HistGradientBoostingClassifier(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "adaboost":
        model = train_classifier(AdaBoostClassifier(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "extratrees":
        model = train_classifier(ExtraTreesClassifier(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "linearsvc":
        model = train_classifier(LinearSVC(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "decisiontree":
        model = train_classifier(DecisionTreeClassifier(), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "logisticregression":
        model = train_classifier(LogisticRegression(max_iter=3000), X_train_pca, y_train_encoded)
    elif classifier_name.lower() == "sgdclassifier":
        model = train_classifier(SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=3000), X_train_pca, y_train_encoded)
    else:
        raise ValueError("Unsupported classifier name")

    # Evaluate model and save results
    evaluate_model_and_save(
        model,
        X_test_pca,
        y_test_encoded,
        label_mapping,
        classifier_name=classifier_name.lower(),
        output_folder="./"
    )

if __name__ == "__main__":
    # Take classifier name as user input
    user_input = input("Enter the classifier name (e.g., RandomForest): ")
    main(user_input)
