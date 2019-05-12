
import pandas as pd
import glob
import json
import pickle
import traceback
from lsi import LSI
from bs4 import BeautifulSoup


data_path = r'C:\Users\trist\Documents\Datasets\supreme_court_cases'


def get_data():
    files = glob.glob(data_path + '/*.json')

    data = []
    for i in files:
        if len(data) > 100:
            break

        print(i)
        try:
            with open(i, 'r') as f:
                j = json.load(f)
                if j['html']:

                    soup = BeautifulSoup(j['html'])

                    data.append(soup.text)
        except:
            traceback.print_exc()
    return data


if __name__ == '__main__':
    documents = get_data()
    print(len(documents))

    lsi = LSI()
    lsi.fit(documents)
    print(lsi.predict(documents))




