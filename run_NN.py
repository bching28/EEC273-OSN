import random
import numpy as np
import pandas as pd
import datetime as dt
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder #bryan, new

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

#bryan, new
def one_hot_encoder(combined_df):
    enc = OneHotEncoder(handle_unknown='ignore')
    month_encoded = enc.fit_transform(combined_df['month'].values.reshape(-1,1)).toarray()
    hour_encoded = enc.fit_transform(combined_df['hour'].values.reshape(-1,1)).toarray()

    month_df = pd.DataFrame(month_encoded, columns = ["Month_"+str(int(i)) for i in range(month_encoded.shape[1])])
    combined_df = pd.concat([combined_df, month_df], axis=1)
    hour_df = pd.DataFrame(hour_encoded, columns = ["Hour_"+str(int(i)) for i in range(hour_encoded.shape[1])])
    combined_df = pd.concat([combined_df, hour_df], axis=1)
    combined_df = combined_df.loc[:, (combined_df != 0).any(axis=0)]

    return combined_df

def read_file():
    # code for randomly sampling file:
    # https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame

    print 'Reading in files...'

    # randomly sample spambot file (300,000)
    spam_fp = 'social_spambots_1.csv/tweets.csv'
    n = sum(1 for line in open(spam_fp)) - 1 #number of records in file (excludes header)
    s = 300000 #desired sample size
    skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

    spambots = pd.read_csv(spam_fp, usecols=['retweet_count', 'num_hashtags', 'num_mentions', 'timestamp'], skiprows=skip)

    # randomly sample genuine file (900,000)
    gen_fp = 'genuine_accounts.csv/tweets.csv'
    n = sum(1 for line in open(gen_fp)) - 1 #number of records in file (excludes header)
    s = 900000 #desired sample size
    skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

    genuine = pd.read_csv(gen_fp, usecols=['retweet_count', 'num_hashtags', 'num_mentions', 'timestamp'], skiprows=skip)

    print spambots
    print genuine

    return spambots, genuine

def load_file():
    print 'Loading Pre-Created File...'
    combined_fp = '/Users/bryan/Documents/Classes/UC Davis/Fall Quarter 2018/Network Architecture and Resource Management/Project/combined.csv'
    combined_df = pd.read_csv(combined_fp)
    return combined_df

def read_new_twitter():
    print 'Reading in files...'

    # randomly sample spambot file (300,000)
    legit_fp = '/Users/bryan/Documents/Classes/UC Davis/Fall Quarter 2018/Network Architecture and Resource Management/Project/sample1.txt'
    #n = sum(1 for line in open(legit_fp)) - 1 #number of records in file (excludes header)
    #s = 300000 #desired sample size
    #skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

    legit = pd.read_csv(legit_fp)

    print legit.head()
    print list(legit)



def create_df(spambots, genuine):
    print 'Creating dataframe...'
    spam_df = pd.DataFrame(spambots)
    gen_df = pd.DataFrame(genuine)
    spam_df = spam_df.dropna(axis=0)
    gen_df = gen_df.dropna(axis=0)

    # extract useful information from timestamp
    gen_df['month'] = gen_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
    gen_df['hour'] = gen_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

    spam_df['month'] = spam_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
    spam_df['hour'] = spam_df['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

    # delete timestamp column
    gen_df = gen_df.drop(['timestamp'], axis=1)
    spam_df = spam_df.drop(['timestamp'], axis=1)
    #gen_df = gen_df.drop(['created_at'], axis=1)
    #spam_df = spam_df.drop(['created_at'], axis=1)

    return spam_df, gen_df

def add_ground_truth(spam_df, gen_df):
    print 'Adding ground truth values...'
    spam_df['ground_truth'] = 1
    gen_df['ground_truth'] = 0
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


def output_data(combined_df):
    export_csv = combined_df.to_csv (r'/Users/bryan/Documents/Classes/UC Davis/Fall Quarter 2018/Network Architecture and Resource Management/Project/export_dataframe.csv',
                        index = None, header=True) #Don't forget to add '.csv' at the end of the path

def gephi_output(combined_df):
    gephi_df = combined_df[['user_id', 'in_reply_to_user_id']]
    gephi_df.columns = ['source', 'target']
    export_csv = gephi_df.to_csv (r'/Users/bryan/Documents/Classes/UC Davis/Fall Quarter 2018/Network Architecture and Resource Management/Project/gephi_output.csv',
                        index = None, header=True) #Don't forget to add '.csv' at the end of the path

def main():
    spambots, genuine = read_file()
    spam_df, gen_df = create_df(spambots, genuine)
    spam_df, gen_df = add_ground_truth(spam_df, gen_df)
    #spam_df, gen_df = numerize_data(spam_df, gen_df)
    combined_df = pd.concat([spam_df, gen_df], ignore_index=True) # 1,200,000 samples total

    # for shuffling rows:
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    #combined_df = combined_df.sample(frac=1)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    #combined_df = shuffle(combined_df)

    combined_df = one_hot_encoder(combined_df)

    #output_data(combined_df)

    #combined_df = load_file()
    #gephi_output(combined_df)
    #read_new_twitter()
    X_train, Y_train, X_test, Y_test = data_split(combined_df)
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=41, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	
	
    print X_train.shape
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# Fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=50)
	
	# evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if __name__ == "__main__":
    main()
