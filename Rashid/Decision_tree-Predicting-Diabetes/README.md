<!-- hide -->
# Decision trees - Steep by steep guide
<!-- endhide -->

- Understand a new dataset.
- Process it by applying exploratory data analysis (EDA).
- Model the data using logistic regression.
- Analyze the results and optimize the model.

## 🌱  How to start this project

Follow the instructions below:

1. Create a new repository based on [machine learning project](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) by [clicking here](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Open the newly created repository in Codespace using the [Codespace button extension](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Once the Codespace VSCode has finished opening, start your project by following the instructions below.

## 🚛 How to deliver this project

Once you have finished solving the exercises, be sure to commit your changes, push to your repository and go to 4Geeks.com to upload the repository link.

## 📝 Instructions

### Predicting Diabetes

This dataset originally comes from the National Institute of Diabetes and Digestive and Kidney Diseases. The goal is to predict based on diagnostic measures whether or not a patient has diabetes.

#### Step 1: Loading the dataset

The dataset can be found in this project folder under the name `diabetes.csv`. You can load it into the code directly from the link (`https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv`) or download it and add it by hand in your repository. In this dataset you will find the following variables:

- `Pregnancies`. Number of pregnancies of the patient (numeric)
- `Glucose`. Plasma glucose concentration 2 hours after an oral glucose tolerance test (numeric)
- `BloodPressure`. Diastolic blood pressure (measured in mm Hg) (numeric)
- `SkinThickness`. Triceps skinfold thickness (measured in mm) (numeric)
- `Insulin`. 2-hour serum insulin (measured in mu U/ml) (numeric)
- `BMI`. Body mass index (numeric)
- `DiabetesPedigreeFunction`. Diabetes Pedigree Function (numeric)
- `Age`. Age of patient (numeric)
- `Outcome`. Class variable (0 or 1), being 0 negative in diabetes and 1, positive (numeric)

#### Step 2: Perform a full EDA

This second step is vital to ensure that we keep the variables that are strictly necessary and eliminate those that are not relevant or do not provide information. Use the example Notebook we worked on and adapt it to this use case.

Be sure to conveniently divide the data set into `train` and `test` as we have seen in previous lessons.

#### Step 3: Build a regression model

Start solving the problem by implementing a decision tree and analyze which of the two types satisfies your needs. Train it and analyze its results. Try modifying the function for calculating the purity of the nodes and use all the available ones. Describe them and analyze your results by graphing them.

#### Step 4: Optimize the previous model

After training the tree with the different purity functions, it selects the best of them and optimizes its hyperparameters using a grid search.

#### Step 5: Save the model

Store the model in the corresponding folder.

> NOTE: Solution: https://github.com/4GeeksAcademy/decision-tree-project-tutorial/blob/main/solution.ipynb