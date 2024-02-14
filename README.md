# DNA Splice Junction Classification with Stacking Ensemble

## Overview
This project aims to classify DNA sequences into splice junctions by implementing a stacking ensemble method. Splice junctions are crucial in the process of gene expression, and accurately identifying them can contribute significantly to our understanding of genetic diseases and the functioning of living organisms. The project utilizes a combination of RandomForest and GradientBoosting classifiers as base learners and LogisticRegression as the meta-learner to achieve high accuracy in predictions.

## Dataset
The dataset used in this project consists of DNA sequence data with labeled splice junctions, including Exon-Intron (EI) junctions, Intron-Exon (IE) junctions, and sequences that do not contain a junction (Neither). Each DNA sequence is represented by binary encoded features indicating the presence of specific nucleotides.

## Methodology
The project follows a structured approach to machine learning, starting with data loading and preprocessing, including adjusting class labels and handling missing values. Following preprocessing, the dataset is split into training and test sets to evaluate the model's performance accurately.

A stacking ensemble method is employed for classification, utilizing RandomForest and GradientBoosting as base models to capture different aspects of the data. LogisticRegression serves as the meta-model to combine the base models' predictions effectively.

## Results
The stacking ensemble model achieved an overall accuracy of 96.55%, with high precision, recall, and F1-scores across all classes. This performance indicates the model's effective learning and generalization capabilities in classifying DNA splice junctions. The detailed metrics are as follows:

- **Accuracy:** 96.55%
  
- **Classification Report:**

  | Class | Precision | Recall | F1-Score |
  |-------|-----------|--------|----------|
  | -1    | 0.94      | 0.98   | 0.96     |
  |  0    | 0.94      | 0.94   | 0.94     |
  |  1    | 0.99      | 0.97   | 0.98     |

The model shows particularly strong performance in identifying class 1 with a precision of 0.99 and an F1-score of 0.98, demonstrating its effectiveness in distinguishing between different types of splice junctions.

## Files in the Repository
- `dna_splice_classifier.py`: The main Python script containing the entire pipeline from data preprocessing to model evaluation.

## Dependencies
- Pandas
- scikit-learn
- XGBoost

## Acknowledgments
The dataset used in this project is sourced from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/muhammetvarl/splicejunction-gene-sequences-dataset/data)). Special thanks to the creators and contributors of the used machine learning libraries. A copy of the dataset is also included in the `Docs` folder within this repository for convenience. Special thanks to the creators and contributors of the machine learning libraries used in this project, and to the original dataset providers for making the data publicly accessible for educational and research purposes.
