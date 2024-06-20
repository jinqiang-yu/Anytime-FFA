# importing the relevant modules
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle

def disaster():
    # reading the .tsv file and converting it into a dataframe object
    train_file = '../datasets/disaster-tweets/disaster-tweets_train.csv'
    train = pd.read_csv(train_file)

    # the same is done with the test set 
    test_file = '../datasets/disaster-tweets/disaster-tweets_test.csv'
    test = pd.read_csv(test_file)

    test = test.loc[:, ['text', 'target']] 

    nltk.download('stopwords')
    nltk.download('wordnet')

    # Pre Processing
    stop_words = stopwords.words('english') # creates a list of English stop words
    wnl = WordNetLemmatizer() # I used lemmatizing instead of stemming
    def preprocess(text_column):
        """
        Function:    This function aims to remove links, special 
                     characters, symbols, stop words and thereafter 
                     lemmatise each word in the sentence to transform 
                     the dataset into something more usable for a 
                     machine learning model.
        Input:       A text column
        Returns:     A text column (but transformed)
        """
        new_review = []
        for review in text_column:
           # for every sentence, we perform the necessary pre-processing
           txt = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", 
                         ' ', 
                         str(review).lower()).strip()
           txt = [wnl.lemmatize(i) for i in txt.split(' ') if i not in stop_words]
           new_review.append(' '.join(txt)) # form back into a sentence
        return new_review

    # actually transforming the datasets
    train['text'] = preprocess(train['text'])
    test['text'] = preprocess(test['text'])

    # vectorizing the sentences
    cv = CountVectorizer(binary = True) # implies that it indicates whether the word is present or not.
    cv.fit(train['text']) # find all the unique words from the training set
    #train_x = cv.transform(train_x)
    #test_x = cv.transform(test_x)


    train_x = cv.transform(train.loc[:, 'text'])
    test_x = cv.transform(test.loc[:, 'text'])


    train_x = pd.DataFrame(train_x.A, columns = cv.get_feature_names_out().tolist())
    test_x = pd.DataFrame(test_x.A, columns = cv.get_feature_names_out().tolist())

    final_train = pd.concat([train_x, train.loc[:, 'target']], axis=1)
    final_test = pd.concat([test_x, test.loc[:, 'target']], axis=1)

    final_train.to_csv(train_file.replace('_train', '_process_train'), index=False)
    final_test.to_csv(test_file.replace('_test', '_process_test'), index=False)

def sarcasm():
    dtname = 'sarcasm'

    #split('../datasets/{0}/{0}.csv'.format(dtname), k=5, inccat=False)
    
    # reading the .tsv file and converting it into a dataframe object
    train_file = '../datasets/{0}/{0}_train.csv'.format(dtname)
    train = pd.read_csv(train_file)
    train = train.loc[:, ['headline', 'sarcastic']]

    # the same is done with the test set 
    test_file = train_file.replace('_train.csv', '_test.csv')
    test = pd.read_csv(test_file)

    test = test.loc[:, ['headline', 'sarcastic']] 

    nltk.download('stopwords')
    nltk.download('wordnet')

    # Pre Processing
    stop_words = stopwords.words('english') # creates a list of English stop words
    wnl = WordNetLemmatizer() # I used lemmatizing instead of stemming
    def preprocess(text_column):
        """
        Function:    This function aims to remove links, special 
                     characters, symbols, stop words and thereafter 
                     lemmatise each word in the sentence to transform 
                     the dataset into something more usable for a 
                     machine learning model.
        Input:       A text column
        Returns:     A text column (but transformed)
        """
        new_review = []
        for review in text_column:
           # for every sentence, we perform the necessary pre-processing
           txt = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", 
                         ' ', 
                         str(review).lower()).strip()
           txt = [wnl.lemmatize(i) for i in txt.split(' ') if i not in stop_words]
           new_review.append(' '.join(txt)) # form back into a sentence
        return new_review

    # actually transforming the datasets
    train['headline'] = preprocess(train['headline'])
    test['headline'] = preprocess(test['headline'])

    # vectorizing the sentences
    cv = CountVectorizer(binary = True) # implies that it indicates whether the word is present or not.
    cv.fit(train['headline']) # find all the unique words from the training set

    train_x = cv.transform(train.loc[:, 'headline'])
    test_x = cv.transform(test.loc[:, 'headline'])

    train_x = pd.DataFrame(train_x.A, columns = cv.get_feature_names_out().tolist())
    test_x = pd.DataFrame(test_x.A, columns = cv.get_feature_names_out().tolist())

    final_train = pd.concat([train_x, train.loc[:, 'sarcastic']], axis=1)
    final_test = pd.concat([test_x, test.loc[:, 'sarcastic']], axis=1)

    final_train.to_csv(train_file.replace('_train', '_process_train'), index=False)
    final_test.to_csv(test_file.replace('_test', '_process_test'), index=False)

def mnist():
    for cls in ['1,3', '1,7']:
        for s in ['train', 'test']:
            
            old = '../datasets/mnist/10,10/{}/complete_origin_data.csv.pkl'.format(cls)
            new = '../datasets/mnist/10,10/{}/{}_origin_data.csv.pkl'.format(cls, s)
            os.system('cp {} {}'.format(old, new))
            

            with open(new, 'rb') as f:
                info = pickle.load(f)
            class_names = info['class_names']

            s_file = '../datasets/mnist/10,10/{}/{}_origin.csv'.format(cls, s)
            df = pd.read_csv(s_file)
            class_names = {real_cls: cls_id for cls_id, real_cls in enumerate(class_names)}
            df = df.replace({df.columns[-1]: class_names})
            df.to_csv(s_file.replace('.csv', '_data.csv'), index=False)

if __name__ == '__main__':
    disaster()
    sarcasm()
    mnist()
    exit()
