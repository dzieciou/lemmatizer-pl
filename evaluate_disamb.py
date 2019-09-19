from gensim.models import KeyedVectors
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate

from eval import make_reproducible
from toygger.scoring import word_accuracy_score
from toygger.tagging import MorphDisambiguator
from toygger import nkjp
from xces import load_chunks_set

make_reproducible()

CHUNKS_LIMIT = 10

chunks_X, chunks_y = load_chunks_set(
    'data/train/nkjp/poleval2017/train-analyzed.xml',
    'data/train/nkjp/poleval2017/train-gold.xml',
    limit=100)

nkjp_word2vec = KeyedVectors.load_word2vec_format(
    'data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=5000)

fasttext_word2vec = None

def evaluate(correct_preds):
    scorer = make_scorer(word_accuracy_score)
    md = MorphDisambiguator(nkjp.tagset, nkjp_word2vec, correct_preds=correct_preds)
    # FIXME Bug?
    #       ValueError: Found input variables with inconsistent numbers of samples: [227, 750]
    #       Prawdpodobonie KFold dzieli zle
    cv = cross_validate(md,
                        chunks_X,
                        chunks_y,
                        cv=KFold(4),
                        scoring=scorer)
    print(cv)
    KFold(4).split(chunks_X, chunks_y)

def evaluate_model_only():
    evaluate(correct_preds=False)

def evaluate_model_with_correction():
    evaluate(correct_preds=True)


evaluate_model_only()
