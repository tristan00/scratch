from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA




class LSI():

    def __init__(self):
        pass


    def fit(self, documents, pca_components = 10, num_of_topics = 10, max_df= .1, max_vocab = 1000):
        self.num_of_topics = num_of_topics
        self.max_df = max_df
        self.max_vocab = max_vocab

        self.tfidf_vec = TfidfVectorizer(max_df=self.max_df, max_features = self.max_vocab)
        self.tfidf_vec.fit(documents)
        tfidf_out = self.tfidf_vec.transform(documents).toarray()

        self.pca = PCA(n_components=pca_components)
        pca_out = self.pca.fit_transform(tfidf_out)

        self.kmean = KMeans(n_clusters=self.num_of_topics)
        self.kmean.fit(pca_out)


    def predict(self, documents):
        tfidf_out = self.tfidf_vec.transform(documents).toarray()
        pca_out = self.pca.transform(tfidf_out)
        return self.kmean.predict(pca_out)









