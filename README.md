# SMS Spam Classification Using Support Vector Machine (SVM)

## 📌 Project Overview

This project focuses on classifying SMS messages as **Spam** or **Ham (Legitimate)** using Machine Learning. It uses Natural Language Processing (NLP) techniques to preprocess text messages and a Support Vector Machine (SVM) classifier to predict whether a message is spam or legitimate.

The project includes exploratory data analysis, text preprocessing, TF-IDF feature extraction, model evaluation, hyperparameter tuning, and cross-validation.

## 🎯 Objectives

- Analyze and understand the SMS Spam Collection dataset.
- Preprocess text messages using NLP techniques.
- Convert text into numerical features using TF-IDF.
- Train an SVM model for spam detection.
- Evaluate model performance using standard classification metrics.
- Optimize model parameters using GridSearchCV.
- Predict whether new SMS messages are spam or legitimate.

## 📂 Dataset

**Dataset:** [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset contains SMS messages labeled as:
- **Spam:** Unwanted or unsolicited messages.
- **Ham:** Legitimate messages.

The notebook uses the `v1` and `v2` columns from the dataset, renamed to `label` and `message`, respectively.

Before running the notebook, ensure the dataset is available as `SMS.csv` in the working directory.

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib & Seaborn** — Data visualization
- **NLTK** — Text preprocessing, stopword removal, and lemmatization
- **WordCloud** — Visualization of frequently occurring words
- **Scikit-learn** — Feature extraction, SVM classification, model evaluation, and hyperparameter tuning

## ⚙️ Project Workflow

### 1. Data Loading and Inspection
- Load the SMS dataset.
- Inspect the dataset structure and data types.
- Check for missing values and duplicate messages.
- Examine the distribution of spam and legitimate messages.

### 2. Exploratory Data Analysis (EDA)
- Analyze the distribution of message labels.
- Examine message lengths and word counts.
- Generate word clouds to visualize common words in spam and ham messages.

### 3. Text Preprocessing
The `clean_text()` function performs the following operations:

- Convert text to lowercase.
- Remove URLs and email addresses.
- Remove numeric digits and punctuation.
- Remove English stopwords.
- Apply lemmatization to reduce words to their base forms.
- Remove duplicate SMS messages.

### 4. Train-Test Split
The dataset is divided into training and testing sets using an **80:20 ratio**.

- Training set: 80%
- Testing set: 20%
- Random state: 42
- Stratification: Maintains the class distribution across the two sets.

### 5. Feature Extraction and Model Training
The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text messages into numerical features.

The SVM model is implemented using `LinearSVC` within a Scikit-learn pipeline.

Configuration:
- Maximum TF-IDF features: 5,000
- N-gram range: Unigrams and bigrams `(1, 2)`
- Classifier: Linear Support Vector Classifier (`LinearSVC`)

### 6. Model Evaluation
The model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted spam messages that are actually spam.
- **Recall:** Proportion of actual spam messages correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Classification Report:** Detailed performance metrics for each class.

### 7. Hyperparameter Tuning
`GridSearchCV` with five-fold cross-validation is used to identify the best value of the SVM regularization parameter `C`.

Values tested:

`C = [0.1, 1, 10]`

### 8. Cross-Validation
Five-fold cross-validation is performed to evaluate model consistency. The individual fold scores and mean accuracy are calculated.

### 9. SMS Prediction
A prediction function, `predict_sms()`, is included to classify new SMS messages as spam or ham after preprocessing.

Example messages:
- Spam: "Congratulations! You have won a free iPhone. Click here now."
- Ham: "Hi, are we meeting tomorrow at 3 PM?"

## 🚀 Getting Started

### Prerequisites

Install Python 3 and the required libraries.

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
```

### Installation and Execution

1. Clone the repository:

   ```bash
   git clone <YOUR_REPOSITORY_URL>
   ```

2. Navigate to the project directory:

   ```bash
   cd <YOUR_REPOSITORY_NAME>
   ```

3. Download the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place `SMS.csv` in the project directory.

4. Open `SMS_Spam_SVM.ipynb` in Jupyter Notebook or Google Colab.

5. Run the notebook cells sequentially to preprocess the data, train the model, evaluate its performance, and test SMS predictions.

**Note:** If using Google Colab, upload the dataset when prompted and ensure its filename matches the filename used in the code.

## 📊 Results

The notebook evaluates the trained model using accuracy, precision, recall, F1-score, classification reports, and five-fold cross-validation.

The best SVM hyperparameter is selected using GridSearchCV. Run the notebook to obtain the actual evaluation scores and best parameter values.

## 🔮 Future Improvements

- Compare SVM with Naive Bayes, Logistic Regression, and other classification algorithms.
- Improve text preprocessing and feature engineering.
- Handle class imbalance where necessary.
- Visualize the confusion matrix and compare model performance.
- Develop a simple web application for real-time SMS spam detection.
- Save the trained model for reuse without retraining.

## 👩‍💻 Author

**Your Name**

GitHub: [Your GitHub Profile](<YOUR_GITHUB_PROFILE_URL>)

---

*This project demonstrates the application of Natural Language Processing and supervised Machine Learning for SMS spam detection.*
