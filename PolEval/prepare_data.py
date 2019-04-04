#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs

from xml.sax import handler, make_parser

# 1215513 toks

# <chunk id="ch1">
#  <chunk type="s">
#   <tok>
#    <orth>A</orth>
#    <lex><base>A</base><ctag>brev:pun</ctag></lex>
#    <lex><base>a</base><ctag>conj</ctag></lex>
#    <lex><base>a</base><ctag>interj</ctag></lex>
#    <lex><base>a</base><ctag>prep:nom</ctag></lex>
#    <lex><base>a</base><ctag>qub</ctag></lex>
#    <lex><base>albo</base><ctag>brev:pun</ctag></lex>
#   </tok>
#   ...
#  </chunk>
# </chunk>
  
class XCEShandler(handler.ContentHandler):

    def __init__(self):
        handler.ContentHandler.__init__(self)
        self._sents = []
        self._tagContent = ''

    def startElement(self, name, attrs):
        if name == 'chunk':
            if attrs.get('type', None) == 's':
                self._sents.append([])
        elif name == 'tok':
            # orth + list of interps
             self._sents[-1].append([None, []])
        if name == 'lex':
            self._sents[-1][-1][1].append(['disamb' in attrs, None, None])
                    
    def endElement(self, name):
        if name == 'orth':
            self._sents[-1][-1][0] = self._tagContent
        if name == 'base':
            self._sents[-1][-1][1][-1][1] = self._tagContent
        if name == 'ctag':
            self._sents[-1][-1][1][-1][2] = self._tagContent
        self._tagContent = ''

    def characters(self, content):
        if content.strip():
            self._tagContent += content
    
def extractSents(path):
    parser = make_parser()
    parser.setFeature(handler.feature_external_ges, False)
    parser.setContentHandler(XCEShandler())
    parser.parse(path)
    return parser.getContentHandler()._sents

if __name__ == '__main__':
    analyzed = extractSents('train-analyzed.xml')
    gold = extractSents('train-gold.xml')
    assert (len(analyzed) == len(gold))
    with codecs.open('train.tab', encoding='utf-8', mode='w') as f:
        for sent_i, an in enumerate(analyzed):
            go = gold[sent_i]
            assert (len(an) == len(go))
            for tok_i, tok_an in enumerate(an):
                tok_go = go[tok_i]
                assert (tok_an[0] == tok_go[0])
                orth = tok_an[0]
                tags = set(list(zip(*tok_an[1]))[2])
                assert (len(tok_go[1]) == 1)
                base, tag = tok_go[1][0][1:]
                if 'ign' in tags:
                    assert (tags == set(['ign', tag]))
                    tags = ['ign']
                label = 'IGN' if tags == ['ign'] else ''
                row = (orth, '|'.join(tags), tag, label, '', '')
                print('\t'.join(row), file=f)
            print('EOS\nEOP', file=f)
