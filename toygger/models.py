from tensorflow.python.keras.layers import (
    Input,
    concatenate,
    Masking,
    Bidirectional,
    LSTM,
    Dense
)
from tensorflow.python.keras.models import (
    Model
)
from tensorflow.python.keras.utils import plot_model

HIDDEN_LAYERS = 2
HIDDEN_LAYER_DIMENSION = 384


def create_model(categories, categories_index, word2vec):
    # First dimension of inputs is variable (None) because length of
    # sentences can vary.

    ctags_vec = Input(
        shape=(None, len(categories_index)),
        dtype='float32',
        name='ctags_vec'
    )

    word_vec = Input(
        shape=(None, word2vec.vector_size),
        dtype='float32',
        name='word_vec'
    )

    all_inputs = concatenate([ctags_vec, word_vec])

    masked_inputs = Masking(mask_value=0., )(all_inputs)

    biLSTM = Bidirectional(LSTM(
        HIDDEN_LAYER_DIMENSION,
        return_sequences=True, dropout=0.5)
    )(masked_inputs)

    for _ in range(HIDDEN_LAYERS - 1):
        biLSTM = Bidirectional(LSTM(
            HIDDEN_LAYER_DIMENSION,
            return_sequences=True,
            dropout=0.5)
        )(biLSTM)

    predicted_tag_categories = [
        Dense(
            len(category.values),
            activation='softmax',
            name=f'{category.name}')(biLSTM)
        for category
        in categories
    ]

    model = Model(
        inputs=[ctags_vec, word_vec],
        outputs=predicted_tag_categories
    )

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # TODO Should we remove it? Or enable with some debug flag?
    plot_model(model, to_file='model.png')

    return model
