from sklearn.metrics import accuracy_score


def flatten(chunks):
    return [token.disamb_ctag for chunk in chunks for token in chunk.tokens]

def word_accuracy_score(y, y_pred):
    y = flatten(y)
    y_pred = flatten(y_pred)
    return accuracy_score(y, y_pred)

