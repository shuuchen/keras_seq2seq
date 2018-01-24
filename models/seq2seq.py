from keras.layers import Input, Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from custom_recurrents import AttentionDecoder
from keras.utils import plot_model

# model definition
EMBEDDING_SIZE = 1000
HIDDEN_SIZE = 1000

def seq2seq(maxLen, vocab_size):

    dialogue_input = Input(shape=(maxLen,))

    embedding = Embedding(input_dim=vocab_size,
                          output_dim=EMBEDDING_SIZE,
                          embeddings_initializer='glorot_uniform',
                          input_length=maxLen)(dialogue_input)

    encode1 = LSTM(HIDDEN_SIZE, return_sequences=True)(embedding)
    encode2 = LSTM(HIDDEN_SIZE)(encode1)

    encode2 = RepeatVector(maxLen)(encode2)

    decode1 = LSTM(HIDDEN_SIZE, return_sequences=True)(encode2)
    decode2 = LSTM(HIDDEN_SIZE, return_sequences=True)(decode1)

    dialogue_output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decode2)

    model = Model(inputs=dialogue_input, outputs=dialogue_output)
    #plot_model(model, 'seq2seq.png', show_shapes=True)

    return model

def attention_seq2seq(maxLen, vocab_size):

    dialogue_input = Input(shape=(maxLen,))

    embedding = Embedding(input_dim=vocab_size,
                          output_dim=EMBEDDING_SIZE,
                          embeddings_initializer='glorot_uniform',
                          input_length=maxLen)(dialogue_input)

    encode1 = LSTM(HIDDEN_SIZE, return_sequences=True)(embedding)
    encode2 = LSTM(HIDDEN_SIZE, return_sequences=True)(encode1)

    dialogue_output = AttentionDecoder(HIDDEN_SIZE, vocab_size)(encode2)

    model = Model(inputs=dialogue_input, outputs=dialogue_output)
    #plot_model(model, 'attention_seq2seq.png', show_shapes=True)

    return model

if __name__ == '__main__':
    #model = seq2seq(250, 50)
    model = attention_seq2seq(250, 50)
