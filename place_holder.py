import random
import numpy as np
import pandas as pd
import datetime as dt
import xgboost as xgb
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.decomposition import PCA

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def read_file():
    # code for randomly sampling file:
    # https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame

    print 'Reading in files...'

    # randomly sample spambot file (300,000)
    spam_fp = 'social_spambots_1.csv/tweets.csv'               
    n = sum(1 for line in open(spam_fp)) - 1 #number of records in file (excludes header)
    s = 300000 #desired sample size
    skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list


    spambots = pd.read_csv(spam_fp, usecols=['id', 'user_id', 'in_reply_to_status_id', 'in_reply_to_user_id',
                            'in_reply_to_screen_name', 'retweeted_status_id', 'num_mentions', 'timestamp'], dtype={'place': object},
                            skiprows=skip)

    # randomly sample genuine file (900,000)
    gen_fp = 'genuine_accounts.csv/tweets.csv'
    n = sum(1 for line in open(gen_fp)) - 1 #number of records in file (excludes header)
    s = 900000 #desired sample size
    skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

    genuine = pd.read_csv(gen_fp, usecols=['id', 'user_id', 'in_reply_to_status_id', 'in_reply_to_user_id',
                            'in_reply_to_screen_name', 'retweeted_status_id', 'num_mentions', 'timestamp'], dtype={'id': object, 'place': object},
                            skiprows=skip)

    return spambots, genuine

def load_file():
    print 'Loading Pre-Created File...'
    combined_fp = 'combined.csv'
    combined_df = pd.read_csv(combined_fp)
    return combined_df

def create_df(spambots, genuine):
    print 'Creating dataframe...'
    spam_df = pd.DataFrame(spambots)
    gen_df = pd.DataFrame(genuine)

    # remove troublesome entry
    gen_df = gen_df[gen_df['id'] != 'Fatal error: Maximum execution time of 300 seconds exceeded in /var/www/phpmyadmin/libraries/export/csv.php on line 178']
    gen_df['id'] = gen_df['id'].astype(dtype=int) # convert from type 'object' to type 'int'


    # extract useful information from timestamp
    gen_df['month'] = gen_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
    #gen_df['day'] = gen_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
    gen_df['hour'] = gen_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    #gen_df['minute'] = gen_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)

    spam_df['month'] = spam_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
    #spam_df['day'] = spam_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
    spam_df['hour'] = spam_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    #spam_df['minute'] = spam_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)

    # delete timestamp column
    gen_df = gen_df.drop(['timestamp'], axis=1)
    spam_df = spam_df.drop(['timestamp'], axis=1)

    return spam_df, gen_df

def add_ground_truth(spam_df, gen_df):
    print 'Adding ground truth values...'
    spam_df['ground_truth'] = 1
    gen_df['ground_truth'] = 0
    return spam_df, gen_df

def numerize_data(spam_df, gen_df):
    # 0:  id
    # 1:  user_id
    # 2:  in_reply_to_status_id
    # 3:  in_reply_to_user_id
    # 4:  in_reply_to_screen_name
    # 5:  retweeted_status_id
    # 6:  num_mentions
    # 7:  month
    # 8:  hour
    # 9:  ground_truth

    print 'Assigning numeric values for strings...'

    col_headers = list(spam_df)

    numerize_list = [4, 6]

    for i in numerize_list:
        spam_df[col_headers[i]] = spam_df[col_headers[i]].astype('category')
        spam_df[col_headers[i]] = spam_df[col_headers[i]].cat.codes
        gen_df[col_headers[i]] = gen_df[col_headers[i]].astype('category')
        gen_df[col_headers[i]] = gen_df[col_headers[i]].cat.codes

    return spam_df, gen_df

def data_split(combined_df):
    print 'Splitting train and test data...'
    X = combined_df.drop(['ground_truth'], axis=1)
    X_train = X.iloc[:900000]
    X_test = X.iloc[900000:]

    Y = combined_df['ground_truth']
    Y_train = Y.iloc[:900000].reset_index(drop=True)
    Y_test = Y.iloc[900000:].reset_index(drop=True)

    return X_train, Y_train, X_test, Y_test

def print_feat_import(classifier, X_train):
    print 'Calculating Feature Importances'
    importances = classifier.feature_importances_
    print 'Importances: ', importances
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print colored("Feature ranking:", 'blue')

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

def logistic_regression(X_train, Y_train, X_test, Y_test):
    print colored('\nPERFORMING LOGISTIC REGRESSION', 'red')
    list_solvers = ['liblinear', 'saga', 'lbfgs']

    for s in list_solvers:
        print 'Solver:', s
        classifier = LogisticRegression(solver=s)
        classifier.fit(X_train, Y_train)

        # Determine feature importance
        #print_feat_import(classifier, X_train)

        score = classifier.score(X_test, Y_test)
        print 'Logistic Regression Score:', score

def svr_float(combined_df):

    col_headers = list(combined_df)
    for x in col_headers:
        combined_df[x] = combined_df[x].astype(dtype=np.float64) # convert to type 'float'
    return combined_df

def support_vector_regressor(X_train, Y_train, X_test, Y_test):
    print colored('\nPERFORMING SUPPORT VECTOR REGRESSION', 'red')

    train_col_headers = list(X_train)
    test_col_headers = list(X_test)

    '''
    for x in train_col_headers:
        X_train[x] = X_train[x].astype(dtype=np.float64) # convert to type 'float'
        Y_train[x] = Y_train[x].astype(dtype=np.float64) # convert to type 'float'
    for y in test_col_headers:
        X_test[y] = X_test[y].astype(dtype=np.float64) # convert to type 'float'
        Y_test[y] = Y_test[y].astype(dtype=np.float64) # convert to type 'float'
    '''
    print X_train.dtypes
    print Y_train.dtypes
    print X_test.dtypes
    print Y_test.dtypes

    classifier = SVR(gamma=0.001, C=1.0, epsilon=0.2)
    classifier.fit(X_train, Y_train)

    # Determine feature importance
    #print_feat_import(classifier, X_train)

    score = classifier.score(X_test, Y_test)
    print 'Score:', score

def random_forest(X_train, Y_train, X_test, Y_test):
    print colored('\nPERFORMING RANDOM FOREST', 'red')

    classifier = RandomForestClassifier(n_estimators=2, max_depth=6)
    classifier.fit(X_train, Y_train)

    # Determine feature importance
    #print_feat_import(classifier, X_train)

    score = classifier.score(X_test, Y_test)
    print 'Random Forest Score:', score

def extra_trees(X_train, Y_train, X_test, Y_test):
    print colored('\nPERFORMING EXTRA TREES', 'red')

    for est in range (5, 16):
        for d in range (1,7,1):
            print '\nNum Estimators:', est
            print 'Depth', d
            classifier = ExtraTreesClassifier(n_estimators=est, max_depth=d)
            classifier.fit(X_train, Y_train)

            # Determine feature importance
            #print_feat_import(classifier, X_train)

            score = classifier.score(X_test, Y_test)
            print 'Extra Trees Score', score

def bagged_decision_tree(X_train, Y_train, X_test, Y_test):
    print colored('\nPERFORMING BAGGED DECISION TREES', 'red')
    for est in range (1, 10):
        print '\nNum Estimators:', est
        dt_clf = DecisionTreeClassifier()
        classifier = BaggingClassifier(base_estimator=dt_clf, n_estimators=est, random_state=7)
        classifier.fit(X_train, Y_train)
        score = classifier.score(X_test, Y_test)
        print 'Bagged Decision Tree Score:', score

def xgboost(X_train, Y_train, X_test, Y_test):
    print colored('\nPERFORMING XGBOOST:\n', 'red')

    params = {'max_depth':10, 'eta':0.05, 'silent':1, 'objective':'binary:logistic',
              'min_child_weight':4, 'subsample':0.5,'colsample_bytree':0.7, 'nthread':4}
    classifier = xgb.XGBModel(**params)
    classifier.fit(X_train, Y_train,
                    eval_set=[(X_test, Y_test)],
                    eval_metric='logloss',
                    verbose=True)
    evals_result = classifier.evals_result()
    print evals_result
    #  0.096322, 0.072576, 0.053894, 0.04207, 0.036596, 0.02993, 0.02501, 0.019127, 0.014405
    # min child weight[2,3,4,5,6,7,8,9]: 0.013871, 0.013481, 0.013179, 0.013605, 0.014329, 0.013491, 0.014044, 0.014665

def main():
    #spambots, genuine = read_file()
    #spam_df, gen_df = create_df(spambots, genuine)
    #spam_df, gen_df = add_ground_truth(spam_df, gen_df)
    #spam_df, gen_df = numerize_data(spam_df, gen_df)
    #combined_df = pd.concat([spam_df, gen_df], ignore_index=True) # 1,200,000 samples total

    combined_df = load_file()

    # for shuffling rows:
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    #combined_df = combined_df.sample(frac=1)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    #combined_df = shuffle(combined_df)

    #combined_df = svr_float(combined_df) # for SVR

    X_train, Y_train, X_test, Y_test = data_split(combined_df)

    #support_vector_regressor(X_train, Y_train, X_test, Y_test)
    #logistic_regression(X_train, Y_train, X_test, Y_test)
    random_forest(X_train, Y_train, X_test, Y_test)
    #extra_trees(X_train, Y_train, X_test, Y_test)
    #bagged_decision_tree(X_train, Y_train, X_test, Y_test)
    #xgboost(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()
