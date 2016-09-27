import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

DEFAULT_TRAINING_FILE = 'training.csv'
DEFAULT_SPLIT_RATIO = 0.7


def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(
                            orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def load_data(filename):
    data = pd.read_csv(filename)
    # Random shuffle data (rows)
    data = data.iloc[np.random.permutation(len(data))]
    return data


def preprocess_data(df):
    # Drop columns
    variables_to_drop = ['RefId', 'PurchDate']                    
    df = df.drop(variables_to_drop, axis=1)
    
    # Extract response variable
    y = df['IsBadBuy'].values.tolist()
    df = df.drop('IsBadBuy', axis=1)
    
    # Normalize numerical features
    df1 = df.select_dtypes(include=['int64', 'float64'])
    df1_norm = (df1 - df1.min()) / (df1.max() - df1.min())

    # Onehot encode categorical features
    df2 = df.select_dtypes(exclude=['int64', 'float64'])
    df2_onehot = encode_onehot(df2, df2.columns)
    
    # Combine both dataframes
    frames = [df1_norm, df2_onehot]
    X = pd.concat(frames, axis = 1)
    
    # Impute missing values
    X = X.values
    X = Imputer().fit_transform(X).tolist()
    
    # Return predictor and response variables
    return X, y


def split_data(X, y, ratio):
    boundary = int(len(X)*ratio)
    X_train = X[:boundary]
    X_test = X[boundary:]
    y_train = y[:boundary]
    y_test = y[boundary:]
    return X_train, X_test, y_train, y_test 


def train_model(X_train, y_train, classifier):
    model = classifier.fit(X_train, y_train)
    return model

    
def evaluate_model(y_model, y_test):
    incorrect = 0
    tp = 0
    fp = 0
    fn = 0

    for idx, pred in enumerate(y_model):
        if pred == 1 and y_test[idx] == 1:
            tp += 1
        if pred == 1 and y_test[idx] == 0:
            fp += 1
        if pred == 0 and y_test[idx] == 1:
            fn += 1
        if pred != y_test[idx]:
            incorrect += 1
     
    accuracy = 1-incorrect*1.0/len(y_test)
    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    return accuracy, precision, recall


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def parse_args():
    parser = argparse.ArgumentParser(description='Read training data')
    parser.add_argument('--training_file', '-t', type=str,
                        default=DEFAULT_TRAINING_FILE)
    parser.add_argument('--split_ratio', '-s', type=restricted_float,
                        default=DEFAULT_SPLIT_RATIO)
    args = parser.parse_args()
    return args


def main(args):
    data = load_data(args.training_file)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y, args.split_ratio)

    classifiers = [LogisticRegression(), DecisionTreeClassifier()] 

    # Train and test classifiers
    for classifier in classifiers:
        classifier_name = type(classifier).__name__
        print '-----Training ' + classifier_name  + '------'
        
        model = train_model(X_train, y_train, classifier)
        y_model = model.predict(X_test)
        y_model = y_model.tolist()
        accuracy, precision, recall = evaluate_model(y_model, y_test)
        
        print classifier_name + ' Results:'
        print 'Accuracy: ' + str(accuracy)
        print 'Precision: ' + str(precision)
        print 'Recall: ' + str(recall)


if __name__ == '__main__':
    main(parse_args())	
