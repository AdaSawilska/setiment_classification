import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

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

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)        # Limit to top x features
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()

    # Standardize the features (PCA works better when data is standardized)
    scaler = StandardScaler(with_mean=False)  # with_mean=False to handle sparse data
    X_scaled = scaler.fit_transform(X)
    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=0.99)  # Retain 95% of variance, or you can specify a fixed number like n_components=300
    X_reduced = pca.fit_transform(X_scaled)

    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)

    # Split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_reduced_train, X_reduced_test, y_reduced_train, y_reduced_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)


    model = LogisticRegression()
    model.fit(X_train, y_train)                     # Train a logistic regression model
    y_pred = model.predict(X_test)                  # Make predictions

    model.fit(X_reduced_train, y_reduced_train)
    y_reduced_pred = model.predict(X_reduced_test)

    # Evaluate the model
    print("Accuracy LR:", accuracy_score(y_test, y_pred))
    print("Accuracy RF:", accuracy_score(y_reduced_test, y_reduced_pred))
    print("Confusion Matrix LR:\n", confusion_matrix(y_test, y_pred))
    print("Confusion Matrix RF:\n", confusion_matrix(y_reduced_test, y_reduced_pred))
    print("Classification Report LR:\n", classification_report(y_test, y_pred))
    print("Classification Report RF:\n", classification_report(y_reduced_test, y_reduced_pred))