
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
The AutomatedML Run was created using an instance of AutoMLConfig. The AutoML Config Class is a way of leveraging the AutoML SDK to automate machine learning. The following parameters have been used for the Auto ML Run.

| Parameter        | Value          | Description  |
| :----- |:-----:| :---------------|
| task     | 'classification' | Classification is selected since we are performing binary classification, i.e whether or not a death event occurs |
| debug.log      | 'automl_errors.log"  | The debug information is written to this file instead of the automl.log file |
| training_data | train_data    | train_data is passed that which contains the data to be used for training |
| label_column_name | 'DEATH_EVENT' | Since the DEATH_EVENT column contains what we need to predict, it is passed |
| compute_target | compute_cluster    | The compute target on which we want this AutoML experiment to run is specified |
| experiment_timeout_minutes | 30  | Specifies the time that all iterations combined can take. Due to the lack of resources this is selected as 30 |
| primary_metric | 'accuracy'    | This is the metric that AutoML will optimize for model_selection. Accuracy is selected as it is well suited to problems involving binary classification. |
| enable_earli_stopping | True | Early Stopping is enabled to terminate a run in case the score is not improving in short term. This allows AutoML to explore more better models in less time |
| featurization | 'auto'   | Featurization is set to auto so that the featurization step is done automatically |
| n_cross_validations | 4  | This is specified so that there are 4 different trainings and each training uses 1/4 of data for validation |
| verbosity | logging.INFO   | This specifies the verbosity level for writing to the log file |


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
The model trained using AutoML searched for several algorithms to find which would perform best in this particular use case, several algorithms including LogisticRegression, SVM, Random Forest, MinMaxScaler, MaxAbsScaler, XGBoostClassifier, VotingEnsemble, etc were explored. The algorithm that performed the best was VotingEnsemble with an accuracy of **0.88694**. AutoML automatically selected the best hyperparameters for the model training. AutoML automatically selects the algorithm and associated hyperparameters, the sampling policy, as well as the early stopping policy. It also selects algorithms that are blacklisted or won't work in that particular case (TensorFlowLinearClassifier and TensorFlowDNN in this case)

**The details of the AutoML run can be monitored using the RunDetails Widget**

![Run Details](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Run%20Details.png)


**Once the run was finished the summary of the run can be seen below-**

![Run Completed](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Run%20Summary.png)


**The best model details are shown below-**

![Best Model details](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Best%20Model%20Summary.png)

![Best Model summary](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/AutoML-%20Best%20Model.png)


## Hyperparameter Tuning
The model used for hyperparameter tuning with HyperDrive is a Logistic Regression Model which is trained using a custom coded script- train.py. The dataset is fetched from a url as a TabularDataset. The hyperparameters chosen for the Scikit-learn model are regularization strength (C) and max iterations (max_iter). 

```
"--C": uniform(0.001, 100),
"--max_iter": choice(50, 75, 100, 125, 150)
```

The hyperparameter tuning using HyperDrive requires several steps- Defining parameter search space, defining a sampling method, choosing a primary metric to optimize and selecting an early stopping policy.

The parameter sampling method used for this project is Random Sampling. It randomly selects the best hyperparameters for the model, that way the entire search space does not need to be searched. The random sampling method saves on time and is a lot faster than grid sampling and bayesian sampling which are recommended only if you have budget to explore the entire search space.

The early stopping policy used in this project is Bandit Policy which is based on a slack factor (0.1 in this case) and an evaluation interval (1 in this case). This policy terminates runs where the primary metric is not within the specified slack factor as compared to the best performing run. This would save on time and resources as runs which won't potentially lead to good results would be terminated early.


### Results
The best HyperDrive run achieved an accuracy of **86.67%**. The hyperparameters selected for the best HyperDrive run are listed below- 

| Parameter        | Value          | 
| :----- |:-----:| 
| Regularization Strength (C) | 85.35037 |
| Max iterations (max_iter) | 75 |


**The details of the HyperDrive run are monitored using the Run Details widget.**

![Run Details](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/HyperDrive-%20Run%20Details.png)


The best model obtained from the HyperDrive Experiment achieved an accuracy of **86.67%** The values of the hyperparameters selected for this model are shown below:

![Best Model](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/HyperDrive-%20Best%20Model.png)


## Model Deployment
Since the model trained using AutomatedML achieved a higher accuracy (88.694%), it was chosen for deployment. 

### Steps for Model Deployment

#### Register the Model

```
description = 'AutoML Model trained on heart failure data to predict if death event occurs or not'
tags = None
model = remote_run.register_model(model_name = model_name, description = description, tags = tags)
```


#### Define an Entry Script

The entry script receives data submitted to a deployed web service and passes it to the model. It then takes the response returned by the model and returns that to the client. For an AutoML model this script can be downloaded from files generated by the AutoML run. The following code snippet shows that.

```
script_file_name = 'inference/score.py'
best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'inference/score.py')
```


#### Define an Inference Configuration

An inference configuration describes how to set up the web-service containing your model. It's used later, when you deploy the model.

```
inference_config = InferenceConfig(entry_script=script_file_name)
```


#### Define a Deployment Configuration

```
aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'area': "hfData", 'type': "automl_classification"}, 
                                               description = 'Heart Failure Prediction')
```


#### Deploy the Model

```
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
```


Once the model is deployed the model endpoint can be accessed from the Endpoints sections in the Assets Tab.

![Model Endpoint](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/Model%20Endpoint.png)


The deployment state of the model can be seen as **Healthy** which indicates that the service is healthy and the endpoint is available.

![Healthy State](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/Deployment%20State-%20healthy.png)


Once the model has been deployed, requests were sent to the model. For sending requests to the model the scoring uri as well as the primary key (if authentication is enabled) are required. A post request is created and the format of the data that is needed to be sent can be inferred from the swagger documentation:

![Swagger Documentation](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/Sample%20Request.png)


The following code interacts with the deployed model by sending it 2 data points specified here and in the [data.json](https://github.com/neha7598/azure-ml-capstone/blob/main/data.json) file.

```
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = aci_service.scoring_uri
# If the service is authenticated, set the key or token

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 70.0,
            "anaemia": 1,
            "creatinine_phosphokinase": 4020,
            "diabetes": 1,
            "ejection_fraction": 32,
            "high_blood_pressure": 1,
            "platelets": 234558.23,
            "serum_creatinine": 1.4,
            "serum_sodium": 125,
            "sex": 0,
            "smoking": 1,
            "time": 12
          },
          {
            "age": 75.0,
            "anaemia": 0,
            "creatinine_phosphokinase": 4221,
            "diabetes": 0,
            "ejection_fraction": 22,
            "high_blood_pressure": 0,
            "platelets": 404567.23,
            "serum_creatinine": 1.1,
            "serum_sodium": 115,
            "sex": 1,
            "smoking": 0,
            "time": 7
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

**The result obtained from the deployed service is- **

![Result](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/Result.png)

The requests being sent to the model can be monitored through the Application Insights URL (If Application Insights are enabled) along with failed requests, time taken per request as well as the availability of the deployed service.

![Application Insights](https://github.com/neha7598/azure-ml-capstone/blob/main/screenshots/Application%20Insights.png)


## Screen Recording
The screen recording with the project walkthrough can be seen [here](https://youtu.be/OUlqEsqebtE)


## Future Improvements
Some areas of improvement for future experiments using HyperDrive include selecting different sampling methods and early_stopping policies as well as increasing the number of total runs. Selecting a different sampling method like Grid Sampling (as opposed to Random Sampling in this case) can lead to a more exhaustive search of the search space which can potentially give us a better result. Also, instead of Logistic Regression, the use of other algorithms like Random Fores, XGBoost, etc can be explored.

For AutoML, future experiments can explore having a experiment timeout time of more than 30 minutes, this can lead to a more exhaustive search and potentially better results.
