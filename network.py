# -*- coding: utf-8 -*-

from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, \
    Concatenate, TimeDistributed, Masking, Multiply
from keras.models import Model

from constants import *
from tagset import *


def build_model(indices, settings):
    
    # morphological input
    morph_input = Input(shape=(None, indices.IN_NUMTAGS), dtype='float32')
    inputs = [morph_input]
    concat = [morph_input]
    
    # character biLSTM
    if CHAR_LSTM in settings.VECTOR_REPRS:
        char_input = Input(shape=(settings.SUFFIX_LEN, indices.NUMCHARS), dtype='float32')
        char_biLSTM = Bidirectional(LSTM(64))(char_input)
        char_model = Model(char_input, char_biLSTM)
        total_char_input = Input(shape=(None, settings.SUFFIX_LEN, indices.NUMCHARS), dtype='float32')
        masked_total_char_input = Masking(mask_value=0.,)(total_char_input)
        chars_encoded = TimeDistributed(char_model)(masked_total_char_input)
        inputs.append(total_char_input)
        # mask
        mask_input = Input(shape=(None, 128), dtype='float32')
        inputs.append(mask_input)
        chars_masked = Multiply()([chars_encoded, mask_input])
        concat.append(chars_masked)
    
    # suffix representation
    if SUFFIX in settings.VECTOR_REPRS:
        suffix_input = Input(shape=(None, indices.NUMCHARS * settings.SUFFIX_LEN), dtype='float32')
        inputs.append(suffix_input)
        concat.append(suffix_input)
    
    # suffix embedding
    if SUFFIX_EMBED in settings.VECTOR_REPRS:
        suffix_embed_input = Input(shape=(None,), dtype='float32')
        suffix_embed = Embedding(indices.NUM_SUFFIXES + 2, 64, mask_zero=True)(suffix_embed_input)
        inputs.append(suffix_embed_input)
        concat.append(suffix_embed)
        
    # embedding representation
    if W2V in settings.VECTOR_REPRS:
        embedding_input = Input(shape=(None, settings.WORD2VEC_DIM), dtype='float32')
        inputs.append(embedding_input)
        concat.append(embedding_input)
    
    # Concraft representation
    if CONCRAFT in settings.VECTOR_REPRS:
        L = indices.IN_NUMTAGS + (1 if CONCRAFT_WEIGHTS in settings.VECTOR_REPRS else 0)
        concraft_input = Input(shape=(None, L), dtype='float32')
        inputs.append(concraft_input)
        concat.append(concraft_input)

    # TODO add dropouts?
    
    # concatenate representations
    total_input = concat[0] if len(concat) == 1 else Concatenate()(concat)
    masked_input = Masking(mask_value=0.,)(total_input)
    # bi-LSTM over the resulting representation
    biLSTM = Bidirectional(LSTM(settings.HIDDEN_DIM, return_sequences=True, dropout=0.5))(masked_input)
    for i in range(settings.HIDDEN_LAYERS - 1):
        biLSTM = Bidirectional(LSTM(settings.HIDDEN_DIM, return_sequences=True, dropout=0.5))(biLSTM)
    
    # separate outputs for tag positions
    outputs = []
    for grp in settings.OUTPUT_TAGS:
        outputs.append(Dense(len(TAG_GROUPS[grp]), activation='softmax')(biLSTM))
    
    model = Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

