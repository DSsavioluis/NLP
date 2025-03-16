# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from src.utils import load_movie_review_clean_dataset
from sklearn.naive_bayes import MultinomialNB

corpus = load_movie_review_clean_dataset()
X = corpus['review']
y = corpus['sentiment']
X_train,  X_test,y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=False, stop_words="english")

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)


# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))