




class LDA():
    def __init__(self, num_of_topics = 5):
        self.num_of_topics = num_of_topics

    def fit(self, documents, steps = 10):
        words = list(set())



        # for each epoch
        #put documents in random topics
        #calculate word prod dist
        # adjust document topic membership
