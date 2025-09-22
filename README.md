# Email Spam Detection

This project uses **Machine Learning (Naive Bayes)** to classify emails or SMS messages as **spam** or **ham**. The model is trained and evaluated on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and provides accuracy, confusion matrix, and classification metrics.

---

## Dataset

The dataset is automatically downloaded via [`kagglehub`](https://pypi.org/project/kagglehub/) in the script `spam_detector.py`.

* **Source:** [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* **Format:** CSV file with two columns: `v1` (label: spam/ham) and `v2` (text message).

---

## Features

* Uses **Naive Bayes (MultinomialNB)** classifier.
* Uses **CountVectorizer** to convert text messages into numerical features.
* Splits dataset into **train and test sets** for evaluation.
* Prints **accuracy, confusion matrix, and classification report** after training.
* Fully **automatic dataset download** – no manual CSV upload required.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/email-spam-detection.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Simply run the main script:

```bash
python spam_detector.py
```

You will see output like this:

```
📊 Model Evaluation
Accuracy: 0.98
Confusion Matrix:
[[ ... ]]
Classification Report:
             precision    recall  f1-score   support
...
```

---

## Requirements

* Python 3.x
* pandas
* scikit-learn
* kagglehub

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Folder Structure

```
email-spam-detection/
│
├─ spam_detector.py       # Main Python script
├─ requirements.txt       # Required Python libraries
├─ README.md              # Project description

```

---

## Notes

* You can later extend this project to **take user input messages** for live prediction.
* The project can be a base for **NLP spam detection improvements** like TF-IDF, word embeddings, or deep learning models.

