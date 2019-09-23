from tqdm import tqdm

from lemmatizer.eval import timing
from lemmatizer.morphology import Dictionary


class MorphAnalyzer:
    '''
    Performs morphological analysis on text chunks. Precisely, for each token
    in each chunk provides a list of possible lemmas and their morphological
    categories using a provided dictionary.
    '''

    def __init__(self, dict: Dictionary):
        '''
        Creates morphological analyzer.
        :param dict: morphological dictionary
        '''
        self.dict = dict

    @timing
    def analyze(self, chunks):
        '''
        Updates a list of given chunks with morphological information
        :param chunks: text chunks to update
        '''
        for chunk in tqdm(chunks, desc='Analyzing chunks'):
            for token in chunk.tokens:
                try:
                    entries = self.dict[token.orth]
                    token.ctags = [e.ctag for e in entries]
                    token.lemmas = [e.lemma for e in entries]
                except KeyError:
                    token.ctags = ['ign']
                    token.lemmas = [token.orth]