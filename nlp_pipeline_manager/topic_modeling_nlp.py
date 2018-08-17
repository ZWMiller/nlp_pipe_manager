from nlp_pipeline_manager.nlp_preprocessor import nlp_preprocessor
from nlp_pipeline_manager.nlpipe import nlpipe

class topic_modeling_nlp(nlpipe):
    
    def __init__(self, model, preprocessing_pipeline=None):
        """
        A pipeline for doing supervised nlp. Expects a model and creates
        a preprocessing pipeline if one isn't provided.
        """
        self.model = model
        self._is_fit = False
        if not preprocessing_pipeline:
            self.preprocessor = nlp_preprocessor()
        else:
            self.preprocessor = preprocessing_pipeline
        
    def fit(self, X):
        """
        Trains the vectorizer and model together using the 
        users input training data.
        """
        self.preprocessor.fit(X)
        train_data = self.preprocessor.transform(X)
        self.model.fit(train_data)
        self._is_fit = True
    
    def transform(self, X):
        """
        Makes a prediction on the data provided by the users using the 
        preprocessing pipeline and provided model.
        """
        if not self._is_fit:
            raise ValueError("Must fit the models before transforming!")
        test_data = self.preprocessor.transform(X)
        preds = self.model.transform(test_data)
        return preds
    
    def print_topics(self, num_words=10):
        """
        A function to print out the top words for each topic
        """
        feat_names = self.preprocessor.vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(self.model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feat_names[i]
                                 for i in topic.argsort()[:-num_words - 1:-1]])
            print(message)



if __name__ == "__main__":
    from sklearn import datasets

    categories = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball']
    ng_train = datasets.fetch_20newsgroups(subset='train',
                                           categories=categories,
                                           remove=('headers',
                                                   'footers', 'quotes'))
    ng_train_data = ng_train.data
    ng_train_targets = ng_train.target

    ng_test = datasets.fetch_20newsgroups(subset='test',
                                          categories=categories,
                                          remove=('headers',
                                                  'footers', 'quotes'))
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer

    from topic_modeling_nlp import topic_modeling_nlp


    cv = CountVectorizer(stop_words='english', token_pattern='\\b[a-z][a-z]+\\b')
    cleaning_pipe = nlp_preprocessor(vectorizer=cv)
    topic_chain = topic_modeling_nlp(TruncatedSVD(n_components=15), preprocessing_pipeline=cleaning_pipe)

    topic_chain.fit(ng_train_data)
    topic_chain.print_topics()