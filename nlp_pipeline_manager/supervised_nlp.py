from nlp_pipeline_manager.nlp_preprocessor import nlp_preprocessor
from nlp_pipeline_manager.nlpipe import nlpipe


class supervised_nlp(nlpipe):
    
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
        
    def fit(self, X, y):
        """
        Trains the vectorizer and model together using the 
        users input training data.
        """
        self.preprocessor.fit(X)
        train_data = self.preprocessor.transform(X)
        self.model.fit(train_data, y)
        self._is_fit = True
    
    def predict(self, X):
        """
        Makes a prediction on the data provided by the users using the 
        preprocessing pipeline and provided model.
        """
        if not self._is_fit:
            raise ValueError("Must fit the models before transforming!")
        test_data = self.preprocessor.transform(X)
        preds = self.model.predict(test_data)
        return preds
    
    def score(self, X, y):
        """
        Returns the accuracy for the model after using the trained
        preprocessing pipeline to prepare the data.
        """
        test_data = self.preprocessor.transform(X)
        return self.model.score(test_data, y)


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

    ng_test_data = ng_test.data
    ng_test_targets = ng_test.target

    from sklearn.naive_bayes import MultinomialNB

    nlp_pipe = supervised_nlp(MultinomialNB())
    nlp_pipe.fit(ng_train_data, ng_train_targets)
    print("Accuracy: ", nlp_pipe.score(ng_test_data, ng_test_targets))