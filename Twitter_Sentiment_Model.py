# %% [markdown]
# # Twitter Tweets Sentiment Models

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
from string import punctuation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %%
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# %% [markdown]
# ## import Dataset

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
df = pd.read_csv("/content/drive/MyDrive/Y3S1-cloud/TXSA(G12)/Tweets.csv") #erwin's
#df = pd.read_csv("/content/drive/MyDrive/Y3S1/TXSA-G12/Tweets.csv") #tracy
#df = pd.read_csv("Tweets.csv") #angeline

# %%
# Count the number of entries for each sentiment category
sentiment_counts = df['sentiment'].value_counts()
total_entries = len(df)
target_size = 8000  # Target size of the dataset

# Calculate the ratio for each sentiment category
ratios = sentiment_counts / total_entries

# Calculate the number of entries to sample from each category to maintain the ratio
samples_per_category = (ratios * target_size).round().astype(int)

# Adjust sample sizes if they exceed the available entries
samples_per_category = samples_per_category.clip(upper=sentiment_counts)

# Mark entries for keeping
df['keep'] = False
for sentiment, sample_size in samples_per_category.items():
    indices_to_keep = df[df['sentiment'] == sentiment].sample(sample_size, random_state=42).index
    df.loc[indices_to_keep, 'keep'] = True

# Filter the dataframe to keep only marked entries
df = df[df['keep']].drop('keep', axis=1)

# %% [markdown]
# ## Data Understanding

# %%
df.head()

# %%
df.columns

# %%
df.shape

# %%
df.sentiment.value_counts()

# %%
df.info()

# %%
sns.countplot(x=df['sentiment'])
plt.title('Total Tweets by Sentiment')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.show()

# %%
from collections import Counter
import itertools

# Splitting words in each tweet and flattening the list
all_words = list(itertools.chain(*df['text'].str.split()))

# Counting the frequency of each word
word_freq = Counter(all_words)

# Most common words
most_common_words = word_freq.most_common(20)

# Converting most common words to a DataFrame
most_common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

most_common_words_df

# %%
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(df['text'].astype(str)))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Tweets")
plt.show()

# %%
negative_df = df[df['sentiment'] == 'negative']
positive_df = df[df['sentiment'] == 'positive']
neutral_df = df[df['sentiment'] == 'neutral']

# Generating word clouds for each category
wordcl_negative = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(negative_df['text'].astype(str)))
wordcl_positive = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(positive_df['text'].astype(str)))
wordcl_neutral = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(neutral_df['text'].astype(str)))

# Plotting the word clouds
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(wordcl_negative, interpolation='bilinear')
plt.title('Word Cloud for "Negative" Sentiment')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(wordcl_positive, interpolation='bilinear')
plt.title('Word Cloud for "Positive" Sentiment')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(wordcl_neutral, interpolation='bilinear')
plt.title('Word Cloud for "Neutral" Sentiment')
plt.axis("off")

plt.show()

# %% [markdown]
# ## Data Preparation & Cleaning

# %%
df = df.drop(columns=['textID', 'selected_text'])

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df = df[df['sentiment'] != 'neutral']

# %%
df.loc[:, 'label'] = df['sentiment'].apply(lambda x: 0 if x == 'positive' else 1)

# %%
df.columns

# %%
df.shape

# %%
df.head()

# %%
sns.countplot(x=df['label'])
plt.title('Total Tweets by Sentiment Label')
plt.xlabel('Sentiment Label (0 for Positive, 1 for Negative)')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ### Informal to standard mapping

# %%
informal_to_standard = {
    "b4": "before",
    "u": "you",
    "r": "are",
    "2": "to",
    "4": "for",
    "gr8": "great",
    "l8r": "later",
    "brb": "be right back",
    "btw": "by the way",
    "lol": "laugh out loud",
    "omg": "oh my god",
    "thx": "thanks",
    "pls": "please",
    "idk": "I don't know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "irl": "in real life",
    "jk": "just kidding",
    "k": "okay",
    "np": "no problem",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "ttyl": "talk to you later",
    "ty": "thank you",
    "w/": "with",
    "w/o": "without",
    "y": "why",
    "yr": "your",
    "atm": "at the moment",
    "b/c": "because",
    "bff": "best friends forever",
    "cya": "see you",
    "fyi": "for your information",
    "gg": "good game",
    "gtg": "got to go",
    "hbu": "how about you",
    "idc": "I don't care",
    "ily": "I love you",
    "ilu": "I love you",
    "lmao": "laughing my a** off",
    "nvm": "never mind",
    "ofc": "of course",
    "omw": "on my way",
    "pov": "point of view",
    "qt": "cutie",
    "sup": "what's up",
    "tmi": "too much information",
    "yolo": "you only live once",
    "bruh": "brother",
    "fam": "family",
    "lit": "amazing",
    "noob": "newbie",
    "pwn": "dominate",
    "slay": "do really well",
    "yeet": "to throw",
    "zzz": "sleeping",
    "fml": "f*** my life",
    "asap": "as soon as possible",
    "afk": "away from keyboard",
    "bae": "before anyone else",
    "m8": "mate",
    "sos": "help",
    "sus": "suspicious",
    "wbu": "what about you",
    "yw": "you're welcome",
    "gl": "good luck",
    "hf": "have fun",
    "np": "no problem",
    "ty": "thank you",
    "ttyl": "talk to you later",
    "i`m": "i am",
    "i`ve": "i have",
    "i`ll": "i will",
    "i`d": "i would",
    "you`re": "you are",
    "you`ve": "you have",
    "you`ll": "you will",
    "you`d": "you would",
    "he`s": "he is",
    "he`ll": "he will",
    "he`d": "he would",
    "she`s": "she is",
    "she`ll": "she will",
    "she`d": "she would",
    "it`s": "it is",
    "it`ll": "it will",
    "it`d": "it would",
    "we`re": "we are",
    "we`ve": "we have",
    "we`ll": "we will",
    "we`d": "we would",
    "they`re": "they are",
    "they`ve": "they have",
    "they`ll": "they will",
    "they`d": "they would",
    "that`s": "that is",
    "that`ll": "that will",
    "that`d": "that would",
    "there`s": "there is",
    "there`re": "there are",
    "there`ll": "there will",
    "there`d": "there would",
    "who`s": "who is",
    "who`ll": "who will",
    "who`d": "who would",
    "what`s": "what is",
    "what`re": "what are",
    "what`ll": "what will",
    "what`d": "what would",
    "where`s": "where is",
    "where`ll": "where will",
    "where`d": "where would",
    "when`s": "when is",
    "when`ll": "when will",
    "when`d": "when would",
    "why`s": "why is",
    "why`ll": "why will",
    "why`d": "why would",
    "how`s": "how is",
    "how`ll": "how will",
    "how`d": "how would",
    "n" : "and",
}

# %% [markdown]
# ### Clean Data Original

# %%
def replace_informal(text):
    words = text.split()
    return ' '.join([informal_to_standard.get(word.lower(), word) for word in words])

# %%
df['Clean_Text1'] = df['text'].apply(replace_informal)

# %%
df

# %%
def clean_text(text):
    # Combine punctuation and stopwords into a set for efficient searching
    bad_tokens = set(punctuation).union(set(stopwords.words('english')))

    # Initialize the lemmatizer
    lemma = WordNetLemmatizer()

    # Tokenize and process the text
    tokens = word_tokenize(text.lower())
    clean_tokens = [lemma.lemmatize(t) for t in tokens if t.isalpha() and t not in bad_tokens]

    return ' '.join(clean_tokens)

# %%
df['Clean_Text2'] = df['Clean_Text1'].apply(clean_text)

# %%
df

# %%
nlp = spacy.load("en_core_web_sm")

def clean_text2(text):
    doc = nlp(text)
    cleaned_text = []

    for token in doc:
        # Check if the token is a stop word, punctuation, or not an alpha character
        if not token.is_stop and not token.is_punct and token.is_alpha:
            # Append the lemmatized, lowercase version of the token
            cleaned_text.append(token.lemma_.lower())

    return ' '.join(cleaned_text)

# %%
df['Clean_Text3'] = df['Clean_Text2'].apply(clean_text2)

# %%
df

# %%
def correct_spelling(text):
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    corrected_text = str(TextBlob(text).correct())
    return corrected_text

# %%
df['Clean_Text4'] = df['Clean_Text3'].apply(correct_spelling)

# %%
df

# %%
df2 = df.drop(columns=['text', 'sentiment','Clean_Text1','Clean_Text2','Clean_Text3'])

# %%
#df2.to_csv("/content/drive/MyDrive/Y3S1-cloud/TXSA(G12)/CleanedTweets.csv", index=False) #export the clean data for SAS

# %% [markdown]
# ## Build Model

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score, ConfusionMatrixDisplay, classification_report

pos = df[df['label'] == 0].sample(2000, replace=True)
neg = df[df['label'] == 1].sample(2000, replace=True)

# Concatenate pos and neg labels
df = pd.concat([pos, neg], axis=0)
df.shape

# %%
X = df['Clean_Text4']
y = df['label']

# %%
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Model 1: Support Vector Classifier

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Setting up the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Initialize GridSearchCV with SVC and the parameter grid
grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=1)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Validated Score (accuracy)", grid_search.best_score_)

# %%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Use the best model from GridSearchCV to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Function to visualize model results
def eval(model_name, y_test, y_pred):
    # Generating and displaying a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # Printing classification report
    print(f'Classification Report for {model_name}: \n')
    print(classification_report(y_test, y_pred))

# Evaluate the model
eval('Best Supoort Vector Classifier', y_test, y_pred)

# %% [markdown]
# ### Model 2: Logistic Regression

# %%
# Import necessary libraries and functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# Define the parameter grid for GridSearchCV
param_grid = {
    'random_state': [42],
    'C': [1, 10, 20, 30, 40, 50],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2'],
    'max_iter': [10000]
}

# Initialize the GridSearchCV object with LogisticRegression and the defined param_grid
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the smaller training data subset
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Use the best model to make predictions on the test set for the first configuration
LR_best_model = grid_search.best_estimator_
y_pred = LR_best_model.predict(X_test)

# Evaluate the best model for the first configuration
accuracy = accuracy_score(y_test, y_pred)

# Print the results for the first configuration
print("Best Hyperparameters:", best_params)
print("Best Validated Score (accuracy)", best_score)
print("Accuracy on Test Set:", accuracy)

# %%
# Define a function to visualize and evaluate model performance
def eval(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred) # Calculate the confusion matrix
    t1 = ConfusionMatrixDisplay(cm) # Initialize a display object for the confusion matrix

    # Print a classification report with main classification metrics
    print('Classification Report for Logistic Regression: \n')
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    t1.plot()

# Call the function to display the evaluation results
eval('Classification Report', y_test, y_pred)

# %% [markdown]
# ### Model 3: Random Forest

# %%
model = RandomForestClassifier()
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

#Defining a function to evaluate model results
def eval(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    t1 = ConfusionMatrixDisplay(cm)
    print('Classification Report for Random Forest Classifier: \n')
    print(classification_report(y_test, y_pred))
    t1.plot()
eval('Classification Report', y_test, y_pred)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Set the parameter grid for the first configuration
param_grid_1 = {
    'n_estimators': [200, 250],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
}

# Create the Random Forest classifier
rf_model = RandomForestClassifier()

# Create the GridSearchCV object for the first configuration
grid_search_1 = GridSearchCV(estimator=rf_model, param_grid=param_grid_1, cv=3, scoring='accuracy', n_jobs=-1)

# Perform the grid search on the smaller training data subset
grid_search_1.fit(X_train, y_train)

# Get the best parameters and the best cross-validated score for the first configuration
best_params_1 = grid_search_1.best_params_
best_score_1 = grid_search_1.best_score_

# Use the best model to make predictions on the test set for the first configuration
best_model_1 = grid_search_1.best_estimator_
y_pred_1 = best_model_1.predict(X_test)

# Evaluate the best model for the first configuration
accuracy_1 = accuracy_score(y_test, y_pred_1)

# Print the results for the first configuration
print("Best Hyperparameters (Configuration 1):", best_params_1)
print("Best Validated Score (accuracy) - Configuration 1:", best_score_1)
print("Accuracy on Test Set (Configuration 1):", accuracy_1)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def train_evaluate_random_forest(X, y, n_estimators=250, max_depth=15, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', bootstrap=True):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the RandomForestClassifier with specified hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Define the evaluation function
    def eval(name, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        t1 = ConfusionMatrixDisplay(cm)
        print(f'Classification Report for {name}: \n')
        print(classification_report(y_test, y_pred))
        t1.plot()

    # Call the eval function with the model
    eval('Random Forest Classifier', y_test, y_pred)

    # Return the trained model
    return model

# Trained Random Forest Model:
trained_rf_model = train_evaluate_random_forest(X, y, n_estimators=200, max_depth=15, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=False)

# %% [markdown]
# ## Model Evaluation

# %%
# Function to evaluate and compare models in terms of accuracy and F1 score
def compare_models_metrics(X_test, y_test, models):
    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Print the metrics
        print(f'Model: {name}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print('\n')

# Import necessary library
from sklearn.metrics import f1_score

# Define models to compare
models = {
    'Support Vector Classifier' : best_model,
    'Logistic Regression': LR_best_model,
    'Random Forest Classifier': model
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

# Compare models in terms of accuracy and F1 score
compare_models_metrics(X_test, y_test, models)


