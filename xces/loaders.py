import xml.dom.pulldom as pulldom
from xml.sax import make_parser
from xml.sax.handler import feature_external_ges

import xpath
from smart_open import open
from tqdm import tqdm

from lemmatizer import Token, Chunk


def load_chunks(fpath, limit=None):
    with open(fpath, 'rb') as f:
        events = pulldom.parse(f, parser=_create_parser())
        chunk_id = 0
        chunk_events = _start_events(events, 'chunk')
        for chunk in tqdm(chunk_events, desc=f'Loading chunks from {fpath}'):
            for chunk in _findall(chunk, 'chunk'):
                if chunk_id == limit:
                    return
                chunk_id += 1
                tokens = []
                for tok in _findall(chunk, 'tok'):
                    orth = _findvalue(tok, 'orth')
                    lemmas = []
                    ctags = []
                    disamb_lemma = None
                    disamb_ctag = None
                    for lex in _findall(tok, 'lex'):
                        lemma = _findvalue(lex, 'base')
                        ctag = _findvalue(lex, 'ctag')
                        if lex.getAttribute('disamb') == '1':
                            disamb_lemma = lemma
                            disamb_ctag = ctag
                        else:
                            lemmas.append(lemma)
                            ctags.append(ctag)
                    token = Token(orth, lemmas, ctags, disamb_lemma,
                                  disamb_ctag)
                    tokens.append(token)

                yield Chunk(tokens)

def load_chunks_set(analyzed_fpath, gold_fpath, limit=None):
    analyzed_chunks = load_chunks(analyzed_fpath, limit)
    gold_chunks = load_chunks(gold_fpath, limit)

    # Load into memory
    analyzed_chunks = list(analyzed_chunks)
    gold_chunks = list(gold_chunks)

    for analyzed, gold in zip(analyzed_chunks, gold_chunks):
        if len(analyzed.tokens) != len(gold.tokens):
            raise ValueError('Invalid tokens number')

    return analyzed_chunks, gold_chunks


def merge_chunks(analyzed_fpath, gold_fpath, limit=None):
    analyzed_chunks = load_chunks(analyzed_fpath, limit)
    gold_chunks = load_chunks(gold_fpath, limit)
    for analyzed_chunk, gold_chunk in zip(analyzed_chunks, gold_chunks):
        tokens = []
        if len(analyzed_chunk.tokens) != len(gold_chunk.tokens):
            raise ValueError('Invalid tokens number')
        for analyzed, gold in zip(analyzed_chunk.tokens, gold_chunk.tokens):
            merged = Token(
                analyzed.orth,
                analyzed.lemmas,
                analyzed.ctags,
                gold.disamb_lemma,
                gold.disamb_ctag
            )
            tokens.append(merged)
        yield Chunk(tokens)


def _create_parser():
    parser = make_parser()
    parser.setFeature(feature_external_ges, False)
    return parser


def _start_events(events, tag_name):
    for event, node in events:
        if event == 'START_ELEMENT' and node.tagName in tag_name:
            events.expandNode(node)
            yield node


def _find(root, xpath_expr):
    return xpath.findnode(xpath_expr, root)


def _findall(root, xpath_expr):
    return xpath.find(xpath_expr, root)


def _findvalue(root, xpath_expr):
    return xpath.findvalue(xpath_expr, root)
