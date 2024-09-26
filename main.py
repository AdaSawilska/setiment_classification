import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


# Download NLTK data files (first-time only)
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the dataset
def load_data(path):
    df = pd.read_csv(path)
    print(df.head())
    return df

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)                  # Remove HTML tags
    text = re.sub(r'\W', ' ', text)                     # Remove non-word characters
    text = re.sub(r'\bnot\s(\w+)', r'not_\1', text)     # Handle negations
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)         # Remove isolated letters
    text = re.sub(r'\s+', ' ', text)                    # Remove multiple spaces
    text = text.lower()

    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text


if __name__ == "__main__":

    df = load_data('IMDB Dataset.csv')
    print("Missing values:\n", df.isnull().sum())                   # Check for missing values
    print(df['sentiment'].value_counts())                           # Distribution of positive and negative reviews
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    df['cleaned_review'] = df['review'].apply(preprocess_text)     # Preprocess text and save to new column
    # df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)




    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)        # Limit to top x features
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()
    # X_combined = np.hstack((X, df[['polarity', 'subjectivity']].values))

    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    print("Original shape:", X.shape)


    # Split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, y, df

    param_grid = {
        'C': [0.1, 1, 10],  # Regularization strength
        'solver': ['liblinear']
    }

    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    lr_best_params = grid_search.best_params_
    print("Best parameters for Logistic Regression: ", lr_best_params)


    model = LogisticRegression(
        C=lr_best_params['C'],
        solver=lr_best_params['solver'],
        random_state=42)
    model.fit(X_train, y_train)                     # Train a logistic regression model
    y_pred = model.predict(X_test)                  # Make predictions


    # Evaluate the model
    print("Accuracy LR:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix LR:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report LR:\n", classification_report(y_test, y_pred))
