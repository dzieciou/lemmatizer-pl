from gensim.models import KeyedVectors
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from toygger import nkjp
from toygger.scoring import word_accuracy_score
from toygger.tagging import MorphDisambiguator
from xces import load_chunks_set

word2vecs = [
    'data/cc.pl.300.vec',
    #             'data/nkjp+wiki-forms-all-300-skipg-ns.txt',
    #             'data/nkjp+wiki-lemmas-all-300-skipg-ns.txt'
]

word2vecs = [KeyedVectors.load_word2vec_format(fpath, limit=None)
             for fpath
             in word2vecs]

parameters = {
    'word2vec': word2vecs,
    'bucket_size': [2048],
    'epochs': [2],
}

scorer = make_scorer(word_accuracy_score)

disambiguator = MorphDisambiguator(
    nkjp.tagset,
    word2vecs[0],
    correct_preds=True)

search = GridSearchCV(
    disambiguator,
    parameters,
    cv=10,
    scoring=scorer,
    n_jobs=-1) # all processors

chunks_X, chunks_y = load_chunks_set(
    'data/train/nkjp/poleval2017/train-analyzed.xml',
    'data/train/nkjp/poleval2017/train-gold.xml',
    limit=None)

search.fit(chunks_X, chunks_y)

print(search.best_score_)
print(search.best_params_)
print(search.cv_results_)
