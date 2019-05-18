import gensim
import re
from nltk.corpus import stopwords
stop_words = [str(i).lower() for i in stopwords.words('english')]

class GensimLDA():

    def __init__(self):
        self.num_of_topics = None
        self.max_df = None
        self.max_vocab = None
        self.dictionary = None
        self.corpus = None
        self.lda = None


    def fit(self, documents, num_of_topics = 10, max_df= .1, max_vocab = 1000, passes = 100, min_word_len = 3):
        self.num_of_topics = num_of_topics
        self.max_df = max_df
        self.max_vocab = max_vocab

        documents2 = [str(i).lower() for i in documents]
        documents2 = [re.split(r'[^a-z]+', i) for i in documents2 if i]
        documents3 = []

        for i in documents2:
            new_doc = [j for j in i if len(j) >= min_word_len and j not in stop_words]
            documents3.append(new_doc)

        self.dictionary = gensim.corpora.Dictionary(documents3)
        self.dictionary.save('gensim_temp/text.dict')
        self.corpus = [self.dictionary.doc2bow(text) for text in documents3]
        self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                  id2word=self.dictionary,
                                                  num_topics=self.num_of_topics,
                                                  update_every=1,
                                                  passes=passes)

        print(self.lda.show_topics(num_topics=num_of_topics))


    def predict(self, documents):
        documents = [str(i).lower() for i in documents]
        documents = [re.split(r'[^a-z]+', i) for i in documents if i]

        other_corpus = [self.dictionary.doc2bow(i) for i in documents]
        res = self.lda[other_corpus]

        predicted_topics = []
        for i in res:
            sorted_topics = sorted(i, key= lambda  x : x[1])
            if sorted_topics:
                predicted_topics.append(sorted_topics[0][0])
            else:
                predicted_topics.append(0)

        return predicted_topics



