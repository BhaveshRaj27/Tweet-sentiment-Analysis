# Tweet-sentiment-Analysis
1. INTRODUCTION

Twitter boasts 330 million monthly active users, which allows businesses to reach a broad audience and connect
with customers without intermediaries. On the downside, there’s so much information that it’s hard for brands
to quickly detect negative social mentions that could harm their business. This is why social listening, which
involves monitoring conversations on social media platforms, has become a key strategy in social media
marketing. Listening to customers on Twitter allows companies to understand their audience, keep on top of
what’s being said about their brand, and their competitors, and discover new trends in the industry. Twitter
Sentiment Analysis allows a deeper understanding of how your customers feel. It adds an extra layer to the
traditional metrics used to analyze the performance of brands on social media and provides businesses with
powerful opportunities.
Sentiment Analysis deals with identifying and classifying opinions or sentiments expressed in the source text.
Social media is generating a vast amount of sentiment rich data in the form of tweets, status updates, blog posts,
etc. Sentiment analysis of this user-generated data is very useful in knowing the opinion of the crowd. Twitter
sentiment analysis is difficult compared to general sentiment analysis due to the presence of slang words and
misspellings.
Twitter Sentiment Analysis, therefore means, using advanced text mining techniques to analyze the sentiment
of the text (here, tweet) in the form of positive, negative and neutral. It is also known as Opinion Mining, is
primarily for analyzing conversations, opinions, and sharing of views (all in the form of tweets) for deciding
business strategy, political analysis, and also for assessing public actions. Thus, we are using Twitter Sentiment
analysis Tools to analyze opinions in Twitter data that can help companies to understand how people are talking
about their brand. Here, we try to analyze Twitter posts using the Machine Learning approach, as we know that
text analysis with machine learning is simple, fast, and scalable, and can provide consistent results with a high
level of accuracy.

2. OBJECTIVE

Our objective is to detect tweets associated with negative sentiments. From this dataset we classify a tweet as
hate speech if it has racist or sexist tweets associated with it. Thus, our task here is to classify racist and sexist
tweets from other tweets and filter them out. Our model uses Natural Language Processing (NLP) to make
sense of human language, and machine learning to automatically deliver accurate results. Our data is in CSV
format. In computing, a comma-separated values (CSV) file stores tabular data (numbers and text) in plain text.
Each line of the file is a data record. Each record consists of one or more fields, separated by commas.
Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and
label ‘0’ denotes the tweet is not racist /sexist, our objective is to predict the labels on the given test dataset. We
will try to employ different machine learning techniques to classify the tweets as positive or negative.

3.1 Raw Data.

The raw data consists of train.csv file which contains 31962 tweets. We are using pandas library to deal
with raw data.

3.2 Data Pre-Processing.

Here we are removing unwanted information from the data set like twitter handles, punctuations,
numbers, special characters, multiple spaces, etc.

3.3 Feature Extraction.

It is the most important part of the process. In this we are converting human language (English) into
numerical form. So that it is easily understandable to machine. Here we are using Keras Tokenizer to
vectorize our extracted features.

3.4 Detection Algorithm.

In this part of process, we are using three classifying techniques.
 LSTM Networks

3.5 Predicted Results

In this model, the Evaluation metrics we are using is F1-Score. It consists of Precision and Recall.

4. DESCRIPTION

4.1 Loading Data for Data-Preprocessing.

The given twitter dataset consists of train.csv where we have 31962 labeled tweets with labels 0 and 1. We
train and validate on the train.csv file.
In this step, we load our dataset using pandas read_csv function.

4.2 Removing Twitter Handles(@User).

In our analysis we can clearly see that the Twitter handles do not contribute anything significant to solve our
problem. So, it’s better if we remove them in our dataset.

4.3 Removing Punctuation, Numbers, and Special Characters.

Punctuation, numbers and special characters do not help much. It is better to remove them from the text just
as we removed the twitter handles. We will replace everything except characters and single spaces between
words in this step.

4.4 Tokenization.
Now we will tokenize all the cleaned tweets in our dataset. Tokens are individual terms or words, and
tokenization is the process of splitting a string of text into tokens. Here we define the number of max features
as 1600 and use Keras’ Tokenizer to vectorize and convert text into Sequences so the Network can deal with
it as input.

4.5 Applying Machine Learning Models
Here we have used three models.
1. Logistic Regression Model.
2. Decision Tree Model.
3. LSTM (Long Short Term Memory) Networks

Logistic Regression is a classification algorithm used to assign observations to a discrete set of
classes. Unlike linear regression which outputs continuous number values, logistic regression
transforms its output using the logistic sigmoid function to return a probability value which can then
be mapped to two or more discrete classes.

Decision Tree is the most powerful and popular tool for classification and prediction. A Decision
tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each
branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used
in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback
connections. It can not only process single data points (such as images), but also entire sequences

Evaluation metrics used here is F1-Score.
To know about F1 Score we first have to know about Precision and Recall.
 Precision means the percentage of your results which are relevant.
 Recall refers to the percentage of total relevant results correctly classified by your algorithm.
of data. They have widely been used for speech recognition, language modeling, sentiment
analysis and text prediction.

We always face a trade-off situation between Precision and Recall i.e. High Precision gives low Recall and
vice versa.
In most problems, you could either give a higher priority to maximising precision, or recall, depending upon
the problem you are trying to solve. But in general, there is a simpler metric which takes into account both
precision and recall, and therefore, you can aim to maximise this number to make your model better. This
metric is known as F1-score, which is simply the harmonic mean of Precision and Recall.
So, this metric seems much easier and convenient to work with, as you only have to maximize one score,
instead of balancing two separate scores.
By applying Logistic Regression Model, the F1 Score we get is 0.5721 and by using Decision Tree Model we
achieved F1 score of 0.5498. The LSTM model gives the F1 Score of 0.9503.

6. RESULT & CONCLUSION

The F1-score for Logistic Regression Model is 0.5721 and for Decision Tree Model the F1-score is 0.5498.
The F1-score for LSTM Model is 0.9503. Here we accomplished our goal to detect tweets with negative
sentiments. Here we have used F1-Score instead of accuracy as a measuring parameter in order to avoid high
number of false positives and to establish a balance between Precision and Recall.
In our previous work we concluded that Logistic Regression Model is better than Decision Tree Model to
predict the sentiment of particular tweet from the given data. Logistic Regression Model performs better than
Decision Tree Model because decision tree algorithm is susceptible to overfit the training data in case of
classes which are not well- separated, and hence simple linear boundary of Logistic Regression algorithm
generalizes better.
We previously used logistic regression because of its simplicity and robustness to the outliers, but it is not able
to capture the relationship between constituents of a sequential data. Logistic regression is a memory-less
technique and doesn’t fit well with sequential data. We have solved this problem using memory- based
technique LSTM which can depict the relation between the words in the sentence and predict the real thoughts
conveyed by the tweet. technique LSTM is great tool for anything that has a sequence, since the meaning of a
word depends on the ones that preceded it. In our case, LSTM helped in keeping stopwords, thus preventing
loss of information and robustness to the outliers since the sequence length doesn’t get reduced severely in
presence of stopwords.
