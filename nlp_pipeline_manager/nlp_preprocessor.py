from sklearn.feature_extraction.text import CountVectorizer
from nlp_pipeline_manager.nlpipe import nlpipe


class nlp_preprocessor(nlpipe):

    def __init__(self, vectorizer=CountVectorizer(), tokenizer=None, cleaning_function=None,
                 stemmer=None):
        """
        A class for pipelining our data in NLP problems. The user provides a series of 
        tools, and this class manages all of the training, transforming, and modification
        of the text data.
        ---
        Inputs:
        vectorizer: the model to use for vectorization of text data
        tokenizer: The tokenizer to use, if none defaults to split on spaces
        cleaning_function: how to clean the data, if None, defaults to the in built class
        stemmer: a function that returns a stemmed version of a token. For NLTK, this
        means getting a stemmer class, then providing the stemming function underneath it.
        """
        if not tokenizer:
            tokenizer = self.splitter
        if not cleaning_function:
            cleaning_function = self.clean_text
        self.stemmer = stemmer
        self.tokenizer = tokenizer
        self.cleaning_function = cleaning_function
        self.vectorizer = vectorizer
        self._is_fit = False
        
    def splitter(self, text):
        """
        Default tokenizer that splits on spaces naively
        """
        return text.split(' ')
        
    def clean_text(self, text, tokenizer, stemmer):
        """
        A naive function to lowercase all works can clean them quickly.
        This is the default behavior if no other cleaning function is specified
        """
        cleaned_text = []
        for post in text:
            cleaned_words = []
            for word in tokenizer(post):
                low_word = word.lower()
                if stemmer:
                    low_word = stemmer(low_word)
                cleaned_words.append(low_word)
            cleaned_text.append(' '.join(cleaned_words))
        return cleaned_text
    
    def fit(self, text):
        """
        Cleans the data and then fits the vectorizer with
        the user provided text
        """
        clean_text = self.cleaning_function(text, self.tokenizer, self.stemmer)
        self.vectorizer.fit(clean_text)
        self._is_fit = True
        
    def transform(self, text, return_clean_text=False):
        """
        Cleans any provided data and then transforms the data into
        a vectorized format based on the fit function.
        If return_clean_text is set to True, it returns the cleaned
        form of the text. If it's set to False, it returns the
        vectorized form of the data.
        """
        if not self._is_fit:
            raise ValueError("Must fit the models before transforming!")
        clean_text = self.cleaning_function(text, self.tokenizer, self.stemmer)
        if return_clean_text:
            return clean_text
        return self.vectorizer.transform(clean_text)


if __name__ == '__main__':
    from nltk.stem import PorterStemmer

    ps = PorterStemmer()
    corpus = ['Testing the', 'class for', 'good behavior.']
    nlp = nlp_preprocessor(stemmer=ps.stem)
    nlp.fit(corpus)
    print(nlp.transform(corpus,return_clean_text=True))
    print(nlp.transform(corpus).toarray())