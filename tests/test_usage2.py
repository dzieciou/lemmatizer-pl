from gensim.models import KeyedVectors

from eval import timing
from toygger.tagging import MorphDisambiguator
from xces import load_chunks_set

from toygger import nkjp

analyzed='data/train/nkjp/poleval2017/train-analyzed.xml'
gold='data/train/nkjp/poleval2017/train-gold.xml'

with timing('Loading train data'):
    chunks_X, chunks_y = load_chunks_set(analyzed, gold, limit=1)

with timing('Loading word vectors'):
    word2vec = KeyedVectors.load_word2vec_format('data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=5000)

m = MorphDisambiguator(nkjp.tagset, word2vec)
m.load_model('data/disambiguation.h5')
print(m.predict(chunks_X[0:1]))