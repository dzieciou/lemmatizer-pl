from lemmatizer.text import Token, Chunk


def test_serializing():
    token = Token(orth='słowa', lemmas=['słowo', 'słowianin'])
    assert token.__str__() \
           == '{"orth": "słowa", "lemmas": ["słowo", "słowianin"], "ctags": null, ' \
              '"disamb_lemma": null, "disamb_ctag": null}'

    chunk = Chunk(tokens=[token])
    assert chunk.__str__() \
           == '{"tokens": [{"orth": "słowa", "lemmas": ["słowo", "słowianin"], ' \
              '"ctags": null, "disamb_lemma": null, "disamb_ctag": null}]}'