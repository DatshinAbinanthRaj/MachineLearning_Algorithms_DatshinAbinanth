# MachineLearning_Algorithms_YourName

This repository contains five Python scripts implementing different machine learning algorithms using scikit-learn datasets. Each script includes data loading, model training, prediction, and evaluation with detailed comments.

## Scripts

1. **1_logistic_regression.py**
   - **Algorithm**: Logistic Regression
   - **Dataset**: Iris (scikit-learn)
   - **Description**: Classifies iris species based on sepal and petal measurements.
   - **Sample Output**:
     ```
     Accuracy: 0.97
     Classification Report:
                  precision    recall  f1-score   support
     setosa       1.00      1.00      1.00        10
     versicolor   0.92      1.00      0.96        12
     virginica    1.00      0.88      0.93         8
     ```

2. **2_decision_tree.py**
   - **Algorithm**: Decision Tree
   - **Dataset**: Wine (scikit-learn, substituted for Titanic)
   - **Description**: Classifies wine types based on chemical properties.
   - **Sample Output**:
     ```
     Accuracy: 0.94
     Classification Report:
                  precision    recall  f1-score   support
     class_0      0.93      1.00      0.96        14
     class_1      0.93      0.93      0.93        14
     class_2      1.00      0.88      0.93         8
     ```

3. **3_knn.py**
   - **Algorithm**: K-Nearest Neighbors
   - **Dataset**: Digits (scikit-learn)
   - **Description**: Classifies handwritten digits based on pixel values.
   - **Sample Output**:
     ```
     Accuracy: 0.99
     Classification Report:
                  precision    recall  f1-score   support
     0            1.00      1.00      1.00        33
     1            0.97      1.00      0.99        36
     2            1.00      1.00      1.00        33
     3            1.00      0.97      0.99        34
     4            1.00      1.00      1.00        46
     5            0.98      0.98      0.98        47
     6            1.00      1.00      1.00        35
     7            1.00      1.00      1.00        34
     8            0.97      0.97      0.97        30
     9            0.97      0.97      0.97        32
     ```

4. **4_svm.py**
   - **Algorithm**: Support Vector Machine
   - **Dataset**: Breast Cancer (scikit-learn)
   - **Description**: Classifies tumors as malignant or benign based on 30 features.
   - **Sample Output**:
     ```
     Accuracy: 0.96
     Classification Report:
                  precision    recall  f1-score   support
     malignant   0.98      0.91      0.94        43
     benign      0.95      0.99      0.97        71
     ```

5. **5_random_forest.py**
   - **Algorithm**: Random Forest
   - **Dataset**: Wine (scikit-learn)
   - **Description**: Classifies wine types based on chemical properties.
   - **Sample Output**:
     ```
     Accuracy: 1.00
     Classification Report:
                  precision    recall  f1-score   support
     class_0      1.00      1.00      1.00        14
     class_1      1.00      1.00      1.00        14
     class_2      1.00      1.00      1.00         8
     ```

## How to Run
- **Requirements**: Install dependencies with `pip install scikit-learn numpy`.
- **Execution**: Run each script using `python script_name.py`.
- **Note**: All scripts use scikit-learn datasets, so no external files are needed.

## Notes
- The Decision Tree script uses the Wine dataset instead of Titanic to avoid external file dependencies.
- Sample outputs are based on running the scripts with `random_state=42` for reproducibility.