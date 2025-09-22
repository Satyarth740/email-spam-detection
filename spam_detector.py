def email_spam_detection():
    import kagglehub
    import pandas as pd

    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    df = pd.read_csv(f"{path}/spam.csv", encoding='latin-1')[['v1','v2']]
    df.columns = ['label','text']
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_num'], test_size=0.2, random_state=42
    )

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    y_pred = model.predict(X_test_vec)
    print("\n📊 Model Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    email_spam_detection()
