# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

# Model Details
This project uses a scikit-learn RandomForestClassifier to predict whether a person’s annual income is above or below $50,000. The model contains 200 decision trees and uses class_weight="balanced" to reduce the effect of the imbalanced classes. A fixed random_state=42 is used so the training process can be reproduced.

Categorical features are one-hot encoded, while unseen categories during inference are ignored instead of causing an error. The target variable is encoded using a label binarizer, and the numerical features are used without any scaling. After training, the model, encoder, and label binarizer are saved as Joblib files and loaded by the FastAPI application for prediction.

# Intended Use
This model was developed mainly for educational purpose. It demonstrates a complete machine learning workflow including data preprocessing, model training, evaluation and deployment through a REST API.

The predictions should only be considered as an estimate learned from historical census data. It should not be used for hiring, salary decisions, loans, insurance, housing, education, immigration, or any other application where an automated prediction may have significant impact on a person.

# Training Data
The model is trained using the UCI Adult (Census Income) dataset, which contains 32,561 records and 14 input features.

The numerical features are:
* age
* fnlgt
* education-num
* capital-gain
* capital-loss
* hours-per-week

The categorical features are:
* workclass
* education
* marital-status
* occupation
* relationship
* race
* sex
* native-country

The target variable is salary, where approximately 75.9% of the records belong to the <=50K class and 24.1% belong to the >50K class.

Before training, white spaces are removed from the column names and string values. The dataset is then split into 80% training and 20% testing using stratified sampling with random_state=42. The encoder and label binarizer are fitted only on the training data to avoid any data leakage.

# Evaluation Data
The model is evaluated on the held-out 20% test split from the same dataset. Since the evaluation data comes from the same source as the training data, the reported performance mainly reflects in-distribution testing and may not represent the performance on different datasets.

The project also evaluates several demographic slices including Male, Female, White, and Black records. These slice evaluations provide some indication about the model behavior, however they should not be considered as complete fairness evaluation.

# Metrics
The evaluation reports the following binary classification metrics by treating >50K as the positive class.

* Precision – the percentage of predicted >50K records that are actually >50K.
* Recall – the percentage of actual >50K records that the model correctly identifies.
* F1 Score – harmonic mean of precision and recall.
* Accuracy – overall percentage of correctly classified records.

The repository does not include fixed performance values because they should be reproduced by running the project. Execute starter/starter/train_model.py to generate the evaluation metrics, and run starter/starter/eval_model.py to produce the demographic slice results.

Since the dataset is imbalanced, accuracy alone may give misleading impression of the model performance. A classifier could achieve relatively high accuracy by mostly predicting the majority class.

### Model evaluation for sex = Female:
Precision: 0.758, Recall: 0.563, F-beta: 0.646, Accuracy: 0.930

### Model evaluation for sex = Male:
Precision: 0.739, Recall: 0.630, F-beta: 0.680, Accuracy: 0.820

### Model evaluation for race = White:
Precision: 0.743, Recall: 0.626, F-beta: 0.679, Accuracy: 0.849

### Model evaluation for race = Black:
Precision: 0.785, Recall: 0.548, F-beta: 0.646, Accuracy: 0.915

# Ethical Considerations
The Adult Census dataset contains sensitive attributes such as race and sex, together with other features that may indirectly represent socioeconomic status or age. Therefore, the model may learn some historical bias that already exists in the original dataset.

Although slice evaluations are included, they only cover few demographic groups. More detailed fairness analysis across additional groups and combinations of attributes would be required before considering any real-world application.

This model is intended only for educational purposes and should not be used for making important decisions without proper validation and human review.

# Caveats and Recommendations
Some limitations of this model are listed below:

* The Adult dataset is relatively old and mainly represents the U.S. population.
* Performance on the test dataset may not generalize well to current or different populations.
* Unknown categorical values are encoded as zeros, which may reduce prediction quality.
* The model predicts only class labels and does not provide calibrated probabilities.
* Only limited demographic slices are evaluated.

If this project is extended beyond educational use, additional work should include reporting confusion matrices, evaluating fairness across more demographic groups, validating on newer datasets, monitoring data drift, and retraining the model through documented process.
