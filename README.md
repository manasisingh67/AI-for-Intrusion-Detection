# AI-for-Intrusion-Detection

Executive Summary

As technology is rapidly evolving, cyber-attacks are increasing and emerging day by day with new threat vectors which demand innovative and robust solutions for effective intrusion detection. This capstone project focuses on exploring the potential of Artificial Intelligence (AI) to develop an advanced and robust intrusion detection system (IDS). The primary focus of the project is on leveraging the power of generative AI to identify powerful models that yield superior outcomes and outperform traditional methods in detecting anomalies within network environments. The project aims to scrutinize both network traffic and server/system logs to enhance cybersecurity measures against the ever-growing and evolving cyber threats.

Project Overview

Objectives

Explore and implement AI techniques to develop an effective intrusion detection system that can adapt to evolving cyber threats. 
Conduct extensive research to identify and employ AI models that provide superior outcomes for intrusion detection. Highlight the significance of model selection in the success of the IDS.
Develop algorithms capable of analyzing both network traffic and server/system logs, recognizing the importance of a comprehensive approach to identify anomalies and potential security breaches. 

Approach & Methodology

In the initial stages of the project, we embarked on a comprehensive journey of research and experimentation. Our exploration covered a wide spectrum of machine learning models to assess their efficiency in the realm of intrusion detection. The project’s approach considers the application of supervised machine learning. This machine learning technique accompanied by comprehensive performance metrics and detailed analysis of confusion matrices, forms a robust framework for intrusion detection:

Application of Techniques: Implementing supervised machine learning techniques for intrusion detection.

Performance Metrics: Utilizing key performance metrics, including accuracy, precision, recall, and F1-score, to assess the effectiveness of the models.

Confusion Matrices: Analyzing confusion matrices for each model, providing a nuanced understanding of their capacity to distinguish normal traffic from anomalous activities.

We have considered this approach to develop effective and reliable models to safeguard computer systems against security threats. 
The project’s methodology delved on to explore diverse models and techniques to enhance the accuracy of intrusion detection systems, with a focus on addressing the limitations of generative AI. 

Methodology

The key aspect of our project involved applying supervised machine learning techniques and assessing the performance metrics, including accuracy, precision, recall, and F1-score. These metrics were analyzed through the lens of confusion matrices for each model, providing a comprehensive understanding of their effectiveness in distinguishing normal traffic from anomalous activities.
To achieve our key aspect, the development process follows an iterative methodology, including the following major steps:

Data Selection

To evaluate the effectiveness of the intrusion detection system, we utilized two widely recognized labeled datasets: CICIDS2017 for network traffic analysis and CIDDS-001 for log analysis. These datasets were chosen intentionally to maintain an unbalanced distribution, mirroring real-world traffic conditions where normal instances often outnumber anomalies.
CICIDS2017 is employed for network traffic analysis and is known for its comprehensive set of labeled instances, offering a diverse representation of normal and potentially malicious network activities. Meanwhile, CIDDS-001 serves the purpose of log analysis, focusing on identifying patterns within log files indicative of security threats.
Both datasets originally consist of network traffic dumps in tcp dump/PCAP format and log files in various formats. To streamline the analysis, these files are converted into CSV format, a widely used and structured format suitable for data analysis and machine learning applications.

Data Cleaning

A crucial step in the process involves creating dataframes for the loaded dataset using Pandas library in Python to streamline the handling and preprocessing of datasets, enhancing their suitability for subsequent analysis. The dataset is converted into individual data frames which are further concatenated into a unified structure, resulting in a single Dataframe.
Further to this, dataframe undergoes cleaning and preprocessing function. This includes actions of removing leading and trailing spaces in column names, setting negative values to 0 in numeric columns, dropping columns with zero variance, managing infinite values, eliminating duplicate rows, dropping columns with identical values, and standardizing column names by stripping, lowering, and replacing spaces with underscores.
The function returns the modified and cleaned dataframe which enhances the dataset's quality, making it more responsive to subsequent analyses. 
For network traffic analysis:
Initially the original dataset for PCAP file consist of 2830743 x 79 sized dataframe.
Post data cleaning process the size of the data frame reduced to 2520798 x 66.
Initially the original dataset for log file consists of 172838 x 16 sized dataframe.
Post data cleaning process the size of the data frame reduced to 172838 x 11.
This phase encapsulates a robust process for loading, combining, and refining datasets, setting the stage for effective data analysis in the context of intrusion detection system evaluation. 

Data Processing

This phase outlines a set of functions designed for processing data and preparing models for classification tasks, with specific applications tailored for network traffic analysis and log analysis. These functions enhance modularity and readability, facilitating a systematic and organized approach to data preparation.
Data processing for classification follows below mentioned steps:

1. Import needed libraries: The initial step involves importing essential Python libraries, setting the groundwork for subsequent operations. Commonly used libraries for data manipulation, machine learning, and visualization, such as Pandas, Scikit-Learn, and Matplotlib, are included.

2. Split the dataset: A crucial function for preparing classification data is implemented here. The train_test_split function from Scikit-Learn is employed to divide the dataset into training and testing sets. This division is essential for evaluating the model's performance on unseen data.

3. Scale and Transform: The dataset's features undergo normalization using MinMaxScaler. This ensures that all feature values fall within a specified range, preventing certain features from dominating others during model training. The same scaler is applied to both the training and testing sets to maintain consistency.

4. Encode labels: Label encoding is performed on the categorical labels using Scikit-Learn's LabelEncoder. This transforms the class labels into numeric representations, facilitating model training. The function returns both the encoded labels for the training and testing sets and a dictionary mapping the original label names to their corresponding numeric values for network traffic analysis.

5. Print the top instances: To provide a snapshot of the preprocessed data, the function prints the top 50 instances in both the training and testing sets. This aids in visually inspecting the transformed dataset.

6. Apply PCA and plot variance ratio: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset. The function not only transforms the scaled features using the fitted PCA model but also plots the explained variance ratio. This visual representation helps in deciding the optimal number of components to retain based on the amount of variance explained. 
As per the plotted variance graph, for network traffic analysis, 95% of the dataset can be determined by 10 principal features.

Following data processing steps are followed or modified in addition to above mentioned steps for log analysis. 

1. Restructure dataframe: This function specifically addresses the unique characteristics of log analysis. It adds new columns to the dataframe based on the presence of specific flags in the 'Flags' column. These new columns act as binary indicators, capturing whether certain flags are present or not. And the flag column is then deleted from the dataframe.

2. Encode classes: Similar to the network traffic analysis, class labels undergo encoding using LabelEncoder. The encoded class labels for the training and testing sets are returned, along with a dictionary mapping the original class names to their corresponding numeric values.

3. Print the top instances: Akin to network traffic analysis, this function prints the top 50 instances in both the training and testing sets, offering a glimpse into the transformed log analysis dataset.

4. Apply PCA and plot variance ratio: PCA is again applied to reduce dimensionality, and the explained variance ratio is plotted. This step aids in understanding how much information is retained with different numbers of principal components, a critical consideration in log analysis. In this plotted variance graph, 95% of the dataset can be determined by 7 principal features.

This phase promotes modularity by encapsulating functionalities into well-defined functions. The combination of dataset splitting, scaling, encoding, and PCA ensures that the data is processed systematically, setting the stage for effective classification model training. The inclusion of visualization elements, such as printing instances and variance ratio plots, enhances interpretability and aids in making informed decisions during the training and optimization process.

Data Visualization

Data visualization and exploratory data analysis (EDA) is done as a part of this phase using Python's Pandas, Matplotlib, and other relevant libraries focused on visualizing the distribution of labels within the dataset.
The process begins by printing the count of respective label entries, offering a quick numerical overview of the dataset's class distribution. Subsequently, a countplot is generated to visually represent the label distribution, excluding the 'BENIGN'(normal traffic) class. The x-axis is logarithmically scaled for improved visualization, and the resulting countplot is saved as an image file.

Further EDA with PCA 2D Visualization is performed by importing necessary libraries and subsampling the dataset by selecting a 10% sample from each class. Features and labels are extracted, and PCA is applied for dimensionality reduction to 2D, creating a new dataframe for this representation. The 2D PCA projection is then visualized using a scatter plot, differentiated by class with a legend added for clarity.

Additionally, for network traffic analysis, we have also considered a binary class projection where 'ATTACK' is assigned to non-'BENIGN' labels. PCA is applied again for dimensionality reduction to 2D, and a dataframe is created for the binary class 2D PCA representation. A scatter plot is generated, and a legend is added for visual interpretation.

Similarly for log analysis, the focus shifts to visualizing the relationship between each feature in the dataset and the specified class. Box plots (or violin plots) are generated for each feature grouped by class, providing insights into the distribution of feature values. The number of rows and columns for the subplot grid is calculated, and the layout is adjusted to prevent overlap of subplots. The final visualizations are then displayed.

At the end of this phase, from understanding label distributions to employing PCA for dimensionality reduction and visualizing feature-class relationships the provided guidance facilitates a thorough analysis of the dataset, contributing to informed decision-making in various domains such as intrusion detection system evaluation.

Training, Testing and Optimization

This phase outlines a systematic approach to training, testing, and optimizing machine learning models for classifying tasks, with a particular emphasis on both network traffic analysis and log analysis. This process is crucial for developing models capable of accurately predicting and classifying instances based on given features.

Training and Testing the Model

In the initial stage, various machine learning models are trained on an 80% subset of the dataset designated for training. This adherence to a standard training split ensures that models undergo a comprehensive learning process. Following this training phase, the models are tested on the remaining 20% of the dataset to conduct an initial analysis. The first 50 instances of traffic are printed to facilitate a visual comparison between the actual labels and those predicted by the models, providing valuable insights into their performance.

Training a Specified Classifier: A flexible train_classifier function is incorporated which can train a specified classifier on scaled training data. The function ensures modularity and adaptability, supporting the integration of different classifiers based on the requirements. It also includes a mechanism to manage cases where a specific value is not found in the dictionary, providing options to return a default value or raise an exception.
For network traffic analysis, a variety of classifiers such as Random Forest, KNN, Hist Gradient Booster, Ada Booster, Extra Trees, Linear SVC, Decision Tree classifier, Bernoulli NB classifier, Logistic Regression, and SGDClassifier, are trained on scaled data.
For log analysis, a specified classifier (Random Forest) is trained on scaled data. 

Testing a Specified Classifier: The evaluate_model_and_save function is introduced to comprehensively evaluate a classification model. This function assesses the model using various metrics, generates a classification report, visualizes the confusion matrix, and saves these outputs to a file. The inclusion of a timestamp ensures the uniqueness of output filenames.

Evaluation of a Specified Classifier: The confusion matrix is visualized, and the plot is saved. The top 50 instances with actual and predicted labels using label names are also printed, providing detailed insights into the model's predictions.

Optimization

Based on the initial analysis, both data preprocessing and machine learning models undergo optimization process to enhance accuracy, reduce false positives and capture false negatives. This involves adjusting parameters, fine-tuning preprocessing steps, and selecting different algorithms to achieve improved model performance.

This phase encapsulates a structured and iterative process for developing, assessing, and optimizing machine learning models for classification tasks, emphasizing transparency, adaptability, and comprehensive evaluation. The integration of various classifiers and the systematic evaluation approach contribute to the robustness of the network traffic analysis models.

Results

The performance of the AI based intrusion detection system was evaluated based on effectiveness and efficiency. The application of supervised machine learning techniques demonstrated notable improvements in accuracy, precision, recall, and F1-score. The confusion matrices provided valuable insights into the models' ability to distinguish normal traffic from anomalous activities. 

For network traffic analysis:

Various models exhibited a range of performance metrics with the Extra Tree classifier demonstrating the highest performance with accuracy, precision, recall and F-1 score of 99.46%.

The Extra Tree classifier yield the highest performance parameter followed by Decision Tree, KNN and Random Forest. Across these models, encompassing accuracy, precision, recall, and F1-score, which collectively gauge overall correctness, the metrics ranged from 98.96% to 99.46%. 
In contrast, other classifiers exhibited a broader performance range, extending from 81.42% to 97.72%.
This divergence in performance underscores the distinctive capabilities of each model in accurately classifying instances within the dataset.

It is important that individual models generated distinct confusion matrices and performance parameters, highlighting their predictive capabilities. 

For log analysis:

A specific classifier Random Forest was considered for log analysis which resulted in an accuracy of 99.20%. Beyond accuracy, performance metrics such as precision, recall, and F1-score also exhibited a robust consistency, all aligning at 99.20%. This uniformity across multiple metrics underscores the robustness and reliability of the Random Forest classifier in log analysis for intrusion detection.

Our model yielded promising results, indicating notable improvements in intrusion detection accuracy and effectiveness. The application of supervised machine learning techniques showcased advancements in performance metrics, providing a foundation for establishing an intrusion detection system suitable for real-world scenarios.

Challenges and Solutions

Addressing challenges in intrusion detection involves recognizing and overcoming obstacles that may impact the effectiveness of machine learning models. Few of such challenges could be:

Imbalanced Datasets: Imbalanced datasets, where instances of normal traffic significantly outnumber intrusion cases, pose a common hurdle. In our project, we have intentionally used the unbalanced dataset to relate it with real time traffic. 
However, to increase the accuracy this challenge can be mitigated through oversampling techniques or Synthetic Minority Over-sampling Technique (SMOTE). This ensures that the model learns equally from both classes.

Real-time Processing: Real-time processing presents another critical challenge, demanding low-latency detection to swiftly identify and respond to potential threats. We have reduced the impact by optimizing the algorithms and fine-tuning the data preprocessing process.
Optimizing algorithms for efficiency and streamlining model training while maintain balance between model complexity and computational speed may respond to evolving cyber-attacks in real-time.

Future Work

While the current project has yielded significant insights and improvements, there are avenues for future exploration and enhancement:

Deep Learning Architectures: Explore deep learning architectures such as neural networks for improved feature extraction and pattern recognition in network traffic and logs.
Integration with threat intelligence:  Integrating threat intelligence into the system will provide real-time information on emerging threats. This integration will enhance the system's ability to proactively identify and respond to evolving security risks.

Real-time adaptability and Explainability model: Developing models which can achieve real time adaptability, minimize false positives, and enhance responsiveness to evolving cyber-threats, while ensuring the decision-making process remains transparent and easy to comprehend. This is crucial and effective collaboration between AI systems and human operators for timely detection of continuously evolving threats and attack patterns. 

Conclusion

The capstone project has successfully explored the application of Artificial Intelligence for intrusion detection. By focusing on generative AI and employing supervised machine learning techniques, the project has achieved notable improvements in accuracy and effectiveness. The developed intrusion detection system holds great promise for enhancing cybersecurity in the face of evolving cyber threats. Continued research and development will contribute to further advancements in this critical domain.

Acknowledgements

I extend my sincere gratitude to Professor Kevin Cleary for offering me this valuable opportunity and for providing invaluable guidance and support throughout the duration of this capstone project. I would also like to express my heartfelt thanks to Professor Smith for her support with the capstone deliverables.

References

Fox, G.; Boppana, R.V. Detection of Malicious Network Flows with Low Preprocessing Overhead. Network 2022, 2, 628–642. https://doi.org/10.3390/network2040036 
https://www.kaggle.com/code/kartikjaspal/eda-and-classification-for-beginners/input
https://www.kaggle.com/code/adepvenugopal/logs-dataset#Server-Logs
https://icsdweb.aegean.gr/awid/awid3
https://www.kaggle.com/datasets/hassan06/nslkdd
https://www.kaggle.com/datasets/cicdataset/cicids2017
https://edgecast.medium.com/detecting-malicious-traffic-with-machine-learning-1a4ebc80672e
https://github.com/alexamanpreet/Network-Log-and-Traffic-Analysis
https://github.com/sinanw/ml-classification-malicious-network-traffic
https://www.kaggle.com/code/vinesmsuic/malware-detection-using-deeplearning
https://github.com/ole-knf/A-bidirectional-GPT-approach-for-detecting-malicious-network-traffic
https://www.linkedin.com/advice/0/how-do-you-analyze-network-traffic-logs
Rawand Raouf Abdalla, Alaa Khalil Jumaa, Log File Analysis Based on Machine Learning: A Survey, UHD JOURNAL OF SCIENCE AND TECHNOLOGY, 2022
Tarnowska, Katarzyna & Patel, Araav. (2021). Log-Based Malicious Activity Detection Using Machine and Deep Learning. 10.1007/978-3-030-62582-5_23.
M. Nam, S. Park and D. S. Kim, "Intrusion Detection Method Using Bi-Directional GPT for in-Vehicle Controller Area Networks," in IEEE Access, vol. 9, pp. 124931-124944, 2021, doi: 10.1109/ACCESS.2021.3110524.
Simone Guarino, Luca Faramondi, Roberto Setola, Francesco Flammini, May 4, 2021, "A hardware-in-the-loop water distribution testbed (WDT) dataset for cyber-physical security testing", IEEE Dataport, doi: https://dx.doi.org/10.21227/rbvf-2h90.
T. Sowmya, E.A. Mary Anita, A comprehensive review of AI based intrusion detection system, Measurement: Sensors, Volume 28, 2023, 100827, ISSN 2665-9174, https://doi.org/10.1016/j.measen.2023.100827.
Vanin, P., Newe, T., Dhirani, L.L., O’Connell, E., O’Shea, D., Lee, B., & Rao, M. (2022, December 22). Network Intrusion Detection Systems Using Artificial Intelligence/Machine Learning. In Encyclopedia. https://encyclopedia.pub/entry/39074
https://adyraj.medium.com/application-of-ai-in-intrusion-detection-system-9705d2efe050
ChatGPT AI for paraphrasing

