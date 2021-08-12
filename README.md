# Sentiment-Analysis
The logistic regression classification algorithm using scikit-learn and Python to classify movie reviews as either postive or negative.

# Introduction and Importing the Data
Introduction to the data set and the problem overview.


Import essential modules and helper functions from NumPy, Matplotlib, and scikit-learn.

# Transforming Documents into Feature Vectors
Represent text data using the bag-of-words model from natural language processing and information retrieval.

Construct the vocabulary of the bag-of-words model and transform the provided sample sentences into sparse feature vectors.

# Term Frequency-Inverse Document Frequency
In information retrieval and text mining, we often observe words that crop up across our corpus of documents. These words can lead to bad performance during training and test time because they usually don’t contain useful information. 

A useful statistical technique, Term frequency-inverse document frequency (tf-idf), to downweight these class of words in the feature vector representation. The tf-idf is the product of the term frequency and the inverse document frequency.

# Calculated TF-IDF of the Term 'Is'
Manually calculated the tf-idf of an example.

Applied scikit-learn’s TfidfTransformer to convert sample text into a vector of tf-idf values and apply the L2-normalization to it.

# Data Preparation 
Cleaning and pre-processing text data is a vital process in data analysis and especially in natural language processing tasks.

Strip the data set of reviews of irrelevant characters including HTML tags, punctuation, and emojis using regular expressions.

# Tokenization of Documents
Ensures that k-means image compression is performed only on the slider widget's mouse release events.

Repurpose the data preprocessing and k-means clustering logic from previous tasks to operate on images of your choice.

Visualize how the image changes as the number of clusters fed to the k-means algorithm is varied.

# Document Classification Using Logistic Regression
First, split the data into training and test sets of equal size.

Then create a pipeline to build a logistic regression model.

To estimate the best parameters and model, we employ cross-validated grid-search over a parameter grid.

# Load Saved Model from Disk
Although the time it takes to train logistic regression models is very little, estimating the best parameters for our model using GridSearchCV can take hours given the size of our training set. 

In this step,  load a pre-trained model that will later be used to find the best parameter settings, cross validation score, and the test accuracy.
# Model Accuracy
In this final step, we take a look at the best parameter settings, cross-validation score, and how well our model classifies the sentiments of reviews it has never seen before from the test set.
