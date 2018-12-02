import random
import numpy as np
import pandas as pd
import datetime as dt
import keras
from keras.models import Sequential
from keras.layers import Dense

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

    
def main():
    spambots, genuine = read_file()
    spam_df, gen_df = create_df(spambots, genuine)
    spam_df, gen_df = add_ground_truth(spam_df, gen_df)
    spam_df, gen_df = numerize_data(spam_df, gen_df)
    combined_df = pd.concat([spam_df, gen_df], ignore_index=True) # 1,200,000 samples total
    
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)    
    #combined_df = shuffle(combined_df)

    X_train, Y_train, X_test, Y_test = data_split(combined_df)
    
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# Fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=1000)
	
	# evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	
	
if __name__ == "__main__":
    main()
