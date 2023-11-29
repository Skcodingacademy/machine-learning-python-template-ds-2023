-- hide -->
# Naive Bayes - Step by step guide
<!-- endhide -->

- Understand a new dataset.
- Process it by applying exploratory data analysis (EDA).
- Model the data using Naive Bayes.
- Analyze the results and optimize the model.

## 🌱  How to start this project

Follow the instructions below:

1. Create a new repository based on [machine learning project](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) by [clicking here](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Open the newly created repository in Codespace using the [Codespace button extension](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Once the Codespace VSCode has finished opening, start your project by following the instructions below.

## 🚛 How to deliver this project

Once you have finished solving the exercises, be sure to commit your changes, push to your repository and go to 4Geeks.com to upload the repository link.

## 📝 Instructions

### Sentiment analysis

Naive Bayes models are very useful when we want to analyze sentiment, classify texts into topics or recommendations, as the characteristics of these challenges meet the theoretical and methodological assumptions of the model very well.

In this project you will practice with a dataset to create a review classifier for the Google Play store.

#### Step 1: Loading the dataset

The dataset can be found in this project folder under the name `playstore_reviews.csv`. You can load it into the code directly from the link (`https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv`) or download it and add it by hand in your repository. In this dataset you will find the following variables:

- `package_name`. Name of the mobile application (categorical)
- `review`. Comment about the mobile application (categorical)
- `polarity`. Class variable (0 or 1), being 0 a negative comment and 1, positive (numeric).

#### Step 2: Study of variables and their content

In this case, we have only 3 variables: 2 predictors and a dichotomous label. Of the two predictors, we are really only interested in the comment part, since the fact of classifying a comment as positive or negative will depend on its content, not on the application from which it was written. Therefore, the `package_name` variable should be removed.

When we work with text as in this case, it does not make sense to do an EDA, the process is different, since the only variable we are interested in is the one that contains the text. In other cases where the text is part of a complex set with other numeric predictor variables and the prediction objective is different, then it makes sense to apply an EDA.

However, we cannot work with plain text, it must first be processed. This process consists of several steps:

1. Removing spaces and converting the text to lowercase:
```py
df["column"] = df["column"].str.strip().str.lower()
```
2. Divide the dataset into train and test: `X_train`, `X_test`, `y_train`, `y_test`.
3. Transform the text into a word count matrix. This is a way to obtain numerical features from the text. For this, we use the training set to train the transformer and apply it in test:
```py
vec_model = CountVectorizer(stop_words = "english")
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()
```

Once we have finished we will have the predictors ready to train the model.

#### Step 3: Build a naive bayes model

Start solving the problem by implementing a model of which you will have to choose which of the three implementations to use: `GaussianNB`, `MultinomialNB` or `BernoulliNB`, according to what we have studied in the module. Try now to train it with the two other implementations and confirm if the model you have chosen is the right one.

#### Step 4: Optimize the previous model

After training the model in its three implementations, choose the best option and try to optimize its results with a random forest, if possible.

#### Step 5: Save the model

Store the model in the appropriate folder.

#### Step 6: Explore other alternatives

Which other models of the ones we have studied could you use to try to overcome the results of a Naive Bayes? Argue this and train the model.

> NOTE: Solution: https://github.com/4GeeksAcademy/naive-bayes-project-tutorial/blob/main/solution.ipynb