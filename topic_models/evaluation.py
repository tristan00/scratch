
import pandas as pd
import glob
import json
import pickle
import traceback
from lsi import LSI
from common import clean_text
from bs4 import BeautifulSoup

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

data_path = r'C:\Users\trist\Documents\Datasets\supreme_court_cases'


def get_data():
    files = glob.glob(data_path + '/*.json')

    data = []
    for c, i in enumerate(files):
        if c % 1000 == 0 and c > 0:
            print(c, i)
        try:
            with open(i, 'r') as f:
                j = json.load(f)
                if j['html']:
                    soup = BeautifulSoup(j['html'])
                    data.append(clean_text(soup.text))
        except:
            traceback.print_exc()
    return data


def evaluate_using_supervised(documents, labels):
    #TODO: reserch topic model evaluation, this is nto a good way

    train_x, val_x, train_y, val_y = train_test_split(documents, labels)

    tfidf_vec = TfidfVectorizer(max_df=.1, max_features =1000)
    tfidf_vec.fit(train_x)

    train_vec = tfidf_vec.transform(train_x)
    val_vec = tfidf_vec.transform(val_x)

    model = ExtraTreesClassifier()
    model.fit(train_vec, train_y)
    return accuracy_score(model.predict(val_vec), val_y)


if __name__ == '__main__':
    documents = get_data()
    print(len(documents))

    lsi = LSI(num_of_topics = 10)
    lsi.fit(documents)

    print(evaluate_using_supervised(documents, lsi.predict(documents)))




