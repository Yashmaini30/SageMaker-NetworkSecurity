
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
import pandas as pd
import numpy as np
import sklearn
import joblib
import boto3
from io  import StringIO
import pathlib
import argparse
import os

def model_fn(model_dir):
    clf=joblib.load(os.path.join(model_dir,"model.joblib"))
    return clf


if __name__=="__main__":

    print("Extracting argument")
    parser=argparse.ArgumentParser()

    ## hyperparameter

    parser.add_argument("--n_estimators",type=int,default=100)
    parser.add_argument("--max_depth",type=int,default=6)
    parser.add_argument("--min_samples_split",type=int,default=2)
    parser.add_argument("--random_state",type=int,default=42)

    ## Data,model, and output directories
    parser.add_argument("--model-dir",type=str,default=os.environ.get("SM_MODEL_DIR")) ## defined in sagemaker
    parser.add_argument("--train",type=str,default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test",type=str,default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file",type=str,default="train-V-1.csv")
    parser.add_argument("--test-file",type=str,default="test-V-1.csv")


    args,_=parser.parse_known_args()

    print("Sklearn version:",sklearn.__version__)
    print("joblib version",joblib.__version__)

    print("[INFO] reading data")
    print()
    train_df=pd.read_csv(os.path.join(args.train,args.train_file))
    test_df=pd.read_csv(os.path.join(args.test,args.test_file))


    features=list(train_df.columns)
    label=features.pop(-1)

    X_train=train_df[features]
    y_train=train_df[label]

    X_test=test_df[features]
    y_test=test_df[label]

    print("[INFO] training model")
    print()
    model=RandomForestClassifier(n_estimators=args.n_estimators,random_state=args.random_state,
                                 max_depth=args.max_depth,min_samples_split=args.min_samples_split,
                                 n_jobs=-1,verbose=2)
    
    model.fit(X_train,y_train)
    

    print()

    model_path=os.path.join(args.model_dir,"model.joblib")
    joblib.dump(model,model_path)

    print("[INFO] saving model")
    print()

    y_pred_test=model.predict(X_test)
    test_accuracy=accuracy_score(y_test,y_pred_test)
    test_roc=roc_auc_score(y_test,y_pred_test)
    test_rep=classification_report(y_test,y_pred_test)
    test_cm=confusion_matrix(y_test,y_pred_test)

    print("------Metrics------")
    print("Test Accuracy:",test_accuracy)
    print("Test ROC:",test_roc)
    print("Test Report:\n",test_rep)
    print("Test Confusion Matrix:\n",test_cm)
