from gensim.models import KeyedVectors

from eval import timing
from lemmatizer import nkjp
from lemmatizer.disamb import MorphDisambiguator
from xces import load_chunks_set


def test_usage():

    # TODO Use relative paths, this is only to access SSD
    analyzed='c:/data/train/nkjp/poleval2017/train-analyzed.xml'
    gold='c:/data/train/nkjp/poleval2017/train-gold.xml'

    with timing('Loading test data'):
        chunks_X, chunks_y = load_chunks_set(analyzed, gold, limit=8191)

    with timing('Loading word vectors'):
        word2vec = KeyedVectors.load_word2vec_format('c:/data/nkjp+wiki-forms-all-300-skipg-ns.txt')

    m = MorphDisambiguator(nkjp.tagset, word2vec, correct_preds=True)
    m.load_model('data/disambiguation2.h5')

    tokens = 0
    correct_tokens = 0
    pred_chunks = m.predict(chunks_X)
    for pred_chunk, true_chunk in zip(pred_chunks, chunks_y):
        for pred_token, true_token in zip(pred_chunk.tokens, true_chunk.tokens):
            tokens += 1
            correct_tokens += (pred_token.disamb_ctag == true_token.disamb_ctag)

    print(f'{correct_tokens}/{tokens} = {correct_tokens/tokens} ')

test_usage()