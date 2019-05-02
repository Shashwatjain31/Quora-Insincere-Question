# Quora Insincere Question Classification
A problem today for major websites is, how to handle toxic and divisive content?
Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to remove insincere questions, those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
In this Project we developed models that identify and flag insincere questions. With this model, Quora can develop more scalable methods to detect toxic and misleading content.

### Dataset:
The dataset can be downloaded from: https://www.kaggle.com/c/quora-insincere-questions-classification
The train data has 3 columns question id (qid), question_text and target. Target = 0 means sincere question and target = 1 means insincere question.
The test data set has 2 columns- question id and question_text.


## Algorithms Used:

In this we have used 3 approches:
### 1.XGBoost: 
XGBoost is an ensemble classifier which is an implementation of gradient boosted classifiers. The algorithm used by XGBoost is a variant of gradient boosting. In Gradient Boosting, weak learners (like Shallow Decision Trees) iteratively run on the dataset. With each iteration, the next learner learns from its predecessors to predict the errors of the prior models. These models are then added together.
The model uses the gradient descent algorithm to minimize its error. XGBoost, of late, has almost become a silver bullet and is extensively used in competitions. By using weak learners as base estimators, it overcomes overfitting and aggregation reduces the bias of the weak learners. Thus it is able to overcome the bias-variance tradeoff.
#### Implementation: 
XGBoost algorithm is implemented by the XGBoost library. XGBoost is a fast algorithm and provides parallel tree boosting. We have used the XGBClassifier model provided by the XGBoost library. It provides 3 base learners: gbtree, gblinear, and dart.

### 2. Naive Bayes Classifier
Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set.
### 3. Logistic Regression:
Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.


After evaluating differnt approches found out that the best accuracy is achieved by Logistic Regression algorithm.

### Tools Required: 
Anaconda -  Jupyter Notebook

### Setup
We have used the Anaconda distribution for downloading all required packages and creating virtual environment. You can use pip for your purposes. For a proper execution of code, Python version 3.6.6 is needed. Ideally any version above 3.6.6 should also work.
