import gensim
import re
from nltk.corpus import stopwords
stop_words = [str(i).lower() for i in stopwords.words('english')]

class GensimLDA():

    def __init__(self):
        pass


    def fit(self, documents, num_of_topics = 10, max_df= .1, max_vocab = 1000, passes = 10):
        self.num_of_topics = num_of_topics
        self.max_df = max_df
        self.max_vocab = max_vocab

        documents = [str(i).lower() for i in documents]
        documents = [re.split(r'[^A-Z]+', i) for i in documents if i]
        documents2 = []
        for i in documents:
            new_doc = [j for j in i if len(j) > 0]
            documents2.append(new_doc)

        texts = [[token for token in text if token not in stop_words] for text in documents2]
        dictionary = gensim.corpora.Dictionary(texts)
        dictionary.save('/tmp/text.dict')
        corpus = [dictionary.doc2bow(text) for text in documents]
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                              id2word=dictionary,
                                              num_topics=self.num_of_topics,
                                              update_every=1,
                                              passes=passes)
        print(lda.show_topics(num_topics=num_of_topics))




