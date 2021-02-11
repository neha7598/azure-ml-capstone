from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://raw.githubusercontent.com/neha7598/azure-ml-capstone/main/data/heart_failure_clinical_records_dataset.csv"

    path_to_data="https://raw.githubusercontent.com/neha7598/azure-ml-capstone/main/data/heart_failure_clinical_records_dataset.csv"
    ds = TabularDatasetFactory.from_delimited_files(path=path_to_data)

    data = ds.to_pandas_dataframe()
    x=data.drop('DEATH_EVENT',axis=1)
    y=data['DEATH_EVENT']

    # Split data into train and test sets.

    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.20)

    run = Run.get_context(allow_offline=True)

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model,filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()