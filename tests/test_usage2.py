from gensim.models import KeyedVectors

from eval import timing
from toygger.tagging import MorphDisambiguator
from xces import load_chunks_set

from toygger import nkjp

def test_usage():

    analyzed='c:/data/train/nkjp/poleval2017/train-analyzed.xml'
    gold='c:/data/train/nkjp/poleval2017/train-gold.xml'

    with timing('Loading test data'):
        chunks_X, chunks_y = load_chunks_set(analyzed, gold, limit=20)

    with timing('Loading word vectors'):
        word2vec = KeyedVectors.load_word2vec_format('c:/data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=10)

    m = MorphDisambiguator(nkjp.tagset, word2vec, correct_preds=False)
    m.load_model('data/disambiguation.h5')


    print(m.predict(chunks_X[10:11]))
    print(chunks_y[10:11])