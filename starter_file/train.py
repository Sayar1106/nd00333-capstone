from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a

run = Run.get_context()
ws = run.experiment.workspace()
dataset = ws.datasets['hr-analytics']

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop(["enrollee_id", "city"], axis=1, inplace=True)
    gender = pd.get_dummies(x_df.gender, prefix="gender")
    x_df.drop("gender", axis=1, inplace=True)
    x_df.join(gender)

    enrolled_university = pd.get_dummies(x_df.enrolled_university, prefix="enrolled")
    x_df.drop("enrolled_university", axis=1, inplace=True)
    x_df.join(enrolled_university)

    education_level = pd.get_dummies(x_df.education_level, prefix="education_level")
    x_df.drop("education_level", axis=1, inplace=True)
    x_df.join(education_level)

    major_discipline = pd.get_dummies(x_df.major_discipline, prefix="major_discipline")
    x_df.drop("major_discipline", axis=1, inplace=True)
    x_df.join(major_discipline)
    
    experience = pd.get_dummies(x_df.experience, prefix="experience")
    x_df.drop("experience", axis=1, inplace=True)
    x_df.join(experience)

    company_size = pd.get_dummies(x_df.company_size, prefix="company_size")
    x_df.drop("company_size", axis=1, inplace=True)
    x_df.join(company_size)

    company_type = pd.get_dummies(x_df.company_type, prefix="company_type")
    x_df.drop("company_type", axis=1, inplace=True)
    x_df.join(company_type)

    last_new_job = pd.get_dummies(x_df.last_new_job, prefix="last_new_job")
    x_df.drop("last_new_job", axis=1, inplace=True)
    x_df.join(last_new_job)

    y_df = x_df.pop("target")

    return x_df, y_df
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    x, y = clean_data(ds)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/project_model.joblib')

if __name__ == '__main__':
    main()
