# Is the Performance of Artificial Neural Networks Superior to Classical ML Models in Recognizing Handwritten Digits? 
## A Comparative Study Using the MNIST Dataset

---

### Abstract:
This project explores the performance differences between classical machine learning (ML) models and artificial neural networks (ANNs) for recognizing handwritten digits, using the MNIST dataset. We evaluate three classical ML models: Logistic Regression, Decision Tree, and Random Forest, and compare their performance metrics against a Convolutional Neural Network (CNN), which is a popular deep learning architecture for image recognition tasks. The comparison will be based on precision, recall, F1-score, and accuracy. The best-performing classical ML model will undergo hyperparameter tuning using GridSearchCV algorithm, and its optimized performance will be compared with that of the CNN model. This approach aims to identify the most efficient and accurate model for handwritten digit recognition and analyze their relative strengths and weaknesses.

---

### Project Plan-Conspect:

#### 1. Introduction:
- Overview of handwritten digit recognition.
- Importance of the MNIST dataset in ML and AI.
- Research question: Is an ANN (CNN) significantly better than classical ML models?

#### 2. Background:
- Explanation of classical ML models:
  - **Logistic Regression**: A probabilistic classifier that models the likelihood of class membership.
  - **Decision Tree**: A tree-structured model for decision-making.
  - **Random Forest**: An ensemble of decision trees to improve generalization and accuracy.
- Overview of **Artificial Neural Networks**:
  - Focus on **Convolutional Neural Networks (CNNs)**: A neural network architecture specifically designed for image recognition.

#### 3. Dataset:
- Description of the MNIST dataset:
  - 70,000 grayscale images of handwritten digits (28x28 pixels).
  - 60,000 images in the training set, 10,000 in the test set.
  - Explanation of preprocessing steps: normalization, reshaping, etc.

#### 4. Methodology:
- Implementation of each model:
  - **Classical ML models**:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
  - Evaluation of models based on:
    - Precision
    - Recall
    - F1-score
    - Accuracy
  - Selection of the best-performing classical ML model based on the evaluation metrics.
  - Hyperparameter tuning of the selected classical ML model to optimize performance.
  - **Convolutional Neural Network** (ANN-based model).
- Cross-validation techniques and split between training and testing data.

#### 5. Performance Metrics:
- Accuracy
- Precision, Recall, F1-score
- Computational efficiency (e.g., training time, memory usage)
- Comparative analysis between tuned classical ML model and CNN.

#### 6. Results and Analysis:
- Presentation of the results from each classical ML model.
- Results of hyperparameter tuning for the best-performing classical ML model.
- Comparison of optimized classical ML model with CNN.
- Discussion on strengths and limitations of each approach.

#### 7. Conclusion:
- Summary of findings.
- Answering the research question: Does CNN significantly outperform classical ML models?
- Insights from hyperparameter tuning on classical ML models.
- Considerations for practical use cases of CNNs vs classical models in real-world applications.

#### 8. Future Work:
- Suggestions for improving model performance (e.g., advanced hyperparameter tuning, data augmentation).
- Exploration of other datasets or hybrid model approaches.
- Potential advancements in neural network architectures.
