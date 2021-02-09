
# Capstone Project- Machine Learning Engineer with Microsoft Azure

This project has been submitted as part of the Machine Learning Engineer with Microsoft Azure Nanodegree. The aim of the project is to train models using Automated Machine Learning as well as by tuning hyperparameters with Hyperdrive. The best performing model is then deployed as a web service and is interacted with. The following chart highlights all the steps performed.

## Dataset

### Overview
The dataset that has been selected for this project is the [Heart Failure Prediction](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) Dataset from Kaggle. This dataset can be used to predict mortality from heart failure.

### Task
The task performed is the prediction of a possible death event during the follow- up period of a patient. The dataset contains 12 features that can be used to predict mortality from heart failure: 
- age: Age of the patient
- amaemia: Decrease of red blood cells or hemoglobin
- creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
- diabetes: If the patient has diabetes
- ejection_fraction: Percentage of blood leaving the heart at each contraction
- high_blood_pressure: If the patient has hypertension
- platelets: Platelets in the blood (kiloplatelets/mL)
- serum_creatinine: Level of serum creatinine in the blood (mg/dL)
- serum_sodium: Level of serum sodium in the blood (mEq/L)
- sex: Woman or man
- smoking: If the patient smokes or not
- time: Follow-up period (days)

The target column is DEATH_EVENT which tells if the patient deceased during the follow-up period

### Access
The dataset has been downloaded from Kaggle and uploaded to this GitHub repository. The dataset is then accessed as a TabularDataset using the URL of the raw .csv file.
```
path_to_data= "https://raw.githubusercontent.com/neha7598/azure-ml-capstone/main/data/heart_failure_clinical_records_dataset.csv"
data=TabularDatasetFactory.from_delimited_files(path=path_to_data)
```

## Automated ML
The AutomatedML Run was created using an instance of AutoMLConfig with the following:

- The Experiment Type- the task parameter is used to specify the experiment type
- Data source- The training data is specified. The data should be in tabular format.
- Label Column Name- The label column which is 'DEATH_EVENT' is specified
- Number of cross validations- The number of cross valdations to be performed in this case has been specified as 4
- Compute Target
- AutoML settings

```
automl_config = AutoMLConfig(
    task='classification',
    training_data=train_data,
    label_column_name='DEATH_EVENT',
    n_cross_validations=4,
    compute_target=compute_cluster,
    **automl_settings
)
```
```
automl_settings = {
    "enable_early_stopping" : True,
    "experiment_timeout_minutes": 30,
    "featurization": 'auto',
    "primary_metric": 'accuracy',
    "verbosity": logging.INFO
}
```


### Results
The model trained using AutoML searched for several algorithms to find which would perform best in this particular use case, several algorithms including LogisticRegression, SVM, Random Forest, MinMaxScaler, MaxAbsScaler, XGBoostClassifier, VotingEnsemble, etc were explored. The algorithm that performed the best was VotingEnsemble with an accuracy of 0.88694. AutoML automatically selected the best hyperparameters for the model training. AutoML automatically selects the algorithm and associated hyperparameters, the sampling policy, as well as the early stopping policy. It also selects algorithms that are blacklisted or won't work in that particular case (TensorFlowLinearClassifier and TensorFlowDNN in this case)

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
The details of the AutoML run can be monitored using the RunDetails Widget
![Run Details](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Run%20Details.png)

Once the run has finished the summary of the run can be seen below- 
![Run Completed](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Run%20Summary.png)

The best model details are shown below-
![Best Model details](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Best%20Model%20Summary.png)

![Best Model summary](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Best%20Model.png)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
