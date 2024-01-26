# AI-Based Intrusion Detection System

## Overview

As technology advances, cyber threats are evolving, demanding innovative solutions for effective intrusion detection...

## Project Objectives

1. **Explore and implement AI techniques for adaptive intrusion detection.**
2. **Identify and employ AI models with superior outcomes.**
3. **Develop algorithms capable of analyzing both network traffic and server/system logs.**

## Approach & Methodology

### Approach

- Utilizes supervised machine learning techniques.
- Key performance metrics: accuracy, precision, recall, F1-score.
- In-depth analysis using confusion matrices.

### Methodology

1. **Data Selection:**
   - Utilized CICIDS2017 for network traffic and CIDDS-001 for log analysis.
   - Maintained an intentionally unbalanced distribution to mimic real-world conditions.
   - Please note that the dataset is not uploaded on this repository. You can find it on Kaggle.

2. **Data Cleaning:**
   - Employed Pandas for creating, cleaning, and preprocessing data frames.
   - Standardized data in CSV format.

3. **Data Processing:**
   - Functions for splitting, scaling, encoding labels, and applying PCA.
   - Ensured modularity and consistency for both analyses.

## Data Visualization

- Explored label distributions and PCA 2D visualization for network traffic and log analysis.

## Training, Testing, and Optimization

- Systematic training and testing of various classifiers.
- Developed a flexible `train_classifier` function.
- `evaluate_model_and_save` function for comprehensive model evaluation.

## Results

### Network Traffic Analysis

- Various models demonstrated performance metrics ranging from 98.96% to 99.46%.
- Extra Tree classifier exhibited the highest accuracy, precision, recall, and F1-score.

### Log Analysis

- Random Forest classifier achieved an accuracy of 99.20%.
- Consistent performance across precision, recall, and F1-score.

## Challenges and Solutions

- Addressed challenges such as imbalanced datasets through intentional dataset design.
- Mitigated real-time processing impact by optimizing algorithms and fine-tuning preprocessing.

## Future Work

- Explore deep learning architectures for improved feature extraction.
- Integrate threat intelligence for real-time information on emerging threats.
- Develop models for real-time adaptability and explainability.

## Conclusion

This capstone project successfully applies AI for intrusion detection, showcasing significant improvements in accuracy and effectiveness. Continued research and development are crucial for further advancements in this critical domain.

## Acknowledgements

Special thanks to Professor Kevin Cleary for guidance and support throughout the project.

## References

- [Fox, G.; Boppana, R.V. Detection of Malicious Network Flows with Low Preprocessing Overhead. Network 2022, 2, 628–642.](#)
- [Additional References...](#)

## Appendices

- [Data Visualization for Log Analysis](#)
- [Results of Network Traffic Analysis](#)
