import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
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
                            'in_reply_to_screen_name', 'retweeted_status_id', 'place', 'retweet_count', 'reply_count', 'favorite_count',
                            'num_hashtags', 'num_urls', 'num_mentions', 'created_at', 'timestamp'], dtype={'place': object},
                            skiprows=skip)

    # randomly sample genuine file (900,000)
    gen_fp = 'genuine_accounts.csv/tweets.csv'
    n = sum(1 for line in open(gen_fp)) - 1 #number of records in file (excludes header)
    s = 900000 #desired sample size
    skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

    genuine = pd.read_csv(gen_fp, usecols=['id', 'user_id', 'in_reply_to_status_id', 'in_reply_to_user_id',
                            'in_reply_to_screen_name', 'retweeted_status_id', 'place', 'retweet_count', 'reply_count', 'favorite_count',
                            'num_hashtags', 'num_urls', 'num_mentions', 'created_at', 'timestamp'], dtype={'id': object, 'place': object},
                            skiprows=skip)
    if isinstance(genuine, pd.DataFrame):
        print("good to go")

    return spambots, genuine

def create_df(spambots, genuine):
    print 'Creating dataframe...'
    spam_df = pd.DataFrame(spambots)
    gen_df = pd.DataFrame(genuine)
    test = list(gen_df['id'])
    for val in test:
        if is_number(val) == False:
            print val

    # remove troublesome entry
    gen_df = gen_df[gen_df['id'] != 'Fatal error: Maximum execution time of 300 seconds exceeded in /var/www/phpmyadmin/libraries/export/csv.php on line 178']
    gen_df['id'] = gen_df['id'].astype(dtype=int) # convert from type 'object' to type 'int'

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
    # 6:  place
    # 7:  retweet_count
    # 8:  reply_count
    # 9:  favorite_count
    # 10: num_hashtags
    # 11: num_urls
    # 12: num_mentions
    # 13: created_at
    # 14: timestamp
    # 15: ground_truth

    print 'Assigning numeric values for strings...'

    col_headers = list(spam_df)
    #print list(spam_df)
    #print spam_df.dtypes
    #print gen_df.dtypes

    numerize_list = [4, 6, 13, 14]

    for i in numerize_list:
        spam_df[col_headers[i]] = spam_df[col_headers[i]].astype('category')
        spam_df[col_headers[i]] = spam_df[col_headers[i]].cat.codes
        gen_df[col_headers[i]] = gen_df[col_headers[i]].astype('category')
        gen_df[col_headers[i]] = gen_df[col_headers[i]].cat.codes

    return spam_df, gen_df     

def data_split(combined_df):
    print 'Splitting train and test data...'
    X = combined_df.drop(['ground_truth'], axis=1)
    X_train = X.iloc[:500000]
    X_test = X.iloc[500000:]

    Y = combined_df['ground_truth']
    Y_train = Y.iloc[:500000].reset_index(drop=True) 
    Y_test = Y.iloc[500000:].reset_index(drop=True)

    return X_train, Y_train, X_test, Y_test
    

def random_forest(X_train, Y_train, X_test, Y_test):
    print 'PERFORMING RANDOM FOREST:\n'
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5)
    classifier.fit(X_train, Y_train)
    score = classifier.score(X_test, Y_test)
    print score

def xgboost(X_train, Y_train, X_test, Y_test):
    print 'PERFORMING XGBOOST:\N'
    X_train_NP = X_train.values
    Y_train_NP = Y_train.values
    X_test_NP = X_test.values
    Y_test_NP = Y_test.values


    dtrain = xgb.DMatrix(X_train_NP, label=Y_train_NP)
    dtest = xgb.DMatrix(X_test_NP, label=Y_test_NP)

    num_round = 10

    params = {'max_depth':10, 'eta':0.05, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'mae',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}
    classifier = xgb.XGBModel(**params)  
    classifier.fit(X_train, Y_train,
                    eval_set=[(X_train, Y_train), (X_test, Y_test)],
                    eval_metric='logloss',
                    verbose=True)
    evals_result = classifier.evals_result()
    print evals_result

def main():
    spambots, genuine = read_file()
    spam_df, gen_df = create_df(spambots, genuine)
    spam_df, gen_df = add_ground_truth(spam_df, gen_df)
    spam_df, gen_df = numerize_data(spam_df, gen_df)
    combined_df = pd.concat([spam_df, gen_df], ignore_index=True) # 1,200,000 samples total

    # dropping last entry because error:
    #combined_df.drop(combined_df.index[1199998], inplace=True) # # (1,200,000 - 1) samples total

    # for shuffling rows:
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    #combined_df = combined_df.sample(frac=1)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)    
    #combined_df = shuffle(combined_df)

    X_train, Y_train, X_test, Y_test = data_split(combined_df)
    #random_forest(X_train, Y_train, X_test, Y_test)
    xgboost(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()
