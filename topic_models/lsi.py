from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA




class LSI():
    def __init__(self, num_of_topics = 5):
        self.num_of_topics = num_of_topics


    def fit(self, documents, pca_components = 10):
        self.tfidf_vec = TfidfVectorizer()
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



lsi = LSI(num_of_topics = 5)
lsi







