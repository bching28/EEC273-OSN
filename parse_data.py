import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def read_file():
    spambots = pd.read_csv('social_spambots_1.csv/tweets.csv', usecols=['id', 'user_id', 'in_reply_to_status_id', 'in_reply_to_user_id',
                            'in_reply_to_screen_name', 'retweeted_status_id', 'place', 'retweet_count', 'reply_count', 'favorite_count',
                            'num_hashtags', 'num_urls', 'num_mentions', 'created_at', 'timestamp'], dtype={'place': object})

    genuine = pd.read_csv('genuine_accounts.csv/tweets.csv', usecols=['id', 'user_id', 'in_reply_to_status_id', 'in_reply_to_user_id',
                            'in_reply_to_screen_name', 'retweeted_status_id', 'place', 'retweet_count', 'reply_count', 'favorite_count',
                            'num_hashtags', 'num_urls', 'num_mentions', 'created_at', 'timestamp'], dtype={'id': object, 'place': object})
    return spambots, genuine

def create_df(spambots, genuine):
    spam_df = pd.DataFrame(spambots)
    gen_df = pd.DataFrame(genuine)
    return spam_df, gen_df


spambots, genuine = read_file()
spam_df, gen_df = create_df(spambots, genuine)
combined_df = pd.concat([spam_df, gen_df], ignore_index=True)

# for shuffling rows:
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
#combined_df.sample(frac=1)
combined_df = shuffle(combined_df)
