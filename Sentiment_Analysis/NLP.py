import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from helpers.eda import cat_summary
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)


###############################
# TEXT PRE-PROCESSING
###############################

df = pd.read_csv("datasets/df_sub.csv", sep=",")
df.head()
df.info()

###############################
# Normalizing Case Folding
###############################
df['reviewText'] = df['reviewText'].str.lower()

###############################
# Punctuations
###############################
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')

###############################
# Numbers
###############################
df['reviewText'] = df['reviewText'].str.replace('\d', '')

###############################
# Stopwords
###############################
# nltk.download('stopwords')
sw = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".
                                          join(x for x in str(x).
                                               split() if x not in sw))

###############################
# Tokenization
###############################

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

###############################
# Lemmatization
###############################

df['reviewText'] = df['reviewText'].apply(lambda x: " ".
                                          join([Word(word).
                                               lemmatize() for word in x.
                                               split()]))

df['reviewText'].head(10)

###############################
# TEXT VISUALIZATION
###############################

###############################
# Calculation of Term Frequencies
###############################

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).\
    sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.head()
tf.shape
tf["words"].nunique()
tf["tf"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################

text = " ".join(i for i in df.reviewText)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


###############################
# Wordcloud by Templates
###############################

vbo_mask = np.array(Image.open("outputs/tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=vbo_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


wc.to_file("outputs/vbo.png")

###############################
# SENTIMENT ANALYSIS
###############################

df.head()

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")
sia.polarity_scores("I liked this music but it is not good as the other one")

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["reviewText"][0:10].apply(lambda x: "pos" if sia.
                             polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.
                                               polarity_scores(x)["compound"] >
                                               0 else "neg")


df.groupby("sentiment_label")["overall"].mean()


df.head()

# ngram
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden 
göstereceğim. N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını 
gösterir ve feature üretmek için kullanılır"""

TextBlob(a).ngrams(3)


##################
# Count Vectors
##################

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frequency
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X.toarray()


# n-gram frequency
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names()
X2.toarray()


##################
# TF-IDF
##################

# word tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X.toarray()

# n-gram tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X.toarray()


# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["reviewText"],
                                                    df["sentiment_label"],
                                                    random_state=17)

train_x[0:5]
train_y[0:5]

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# Count Vectors
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)
vectorizer.get_feature_names()[0:10]
x_train_count.toarray()

# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# TF-IDF N-Gram Level
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3)).fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

# TF-IDF Characters Level
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char",
                                          ngram_range=(2, 3)).fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)

###############################
# MODELING (SENTIMENT MODELING)
###############################

# TF-IDF Word-Level Logistic Regression
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)
y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()

new_comment = pd.Series("this film is great")
new_comment = pd.Series("look at that shit very bad")
new_comment = pd.Series("it was good but I am sure that it fits me")

new_comment = CountVectorizer().fit(train_x).transform(new_comment)

log_model.predict(new_comment)

random_review = pd.Series(df["reviewText"].sample(1).values)
new_comment = CountVectorizer().fit(train_x).transform(random_review)
log_model.predict(new_comment)


# RandomForestClassifier
# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(x_train_tf_idf_ngram, train_y)
cross_val_score(rf_model, x_test_tf_idf_ngram, test_y, cv=5, n_jobs=-1).mean()

# TF-IDF CHARLEVEL
rf_model = RandomForestClassifier().fit(x_train_tf_idf_chars, train_y)
cross_val_score(rf_model, x_test_tf_idf_chars, test_y, cv=5, n_jobs=-1).mean()

# Count Vectors
rf_model = RandomForestClassifier().fit(x_train_count, train_y)
cross_val_score(rf_model, x_test_count, test_y, cv=5).mean()
