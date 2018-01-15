from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import plot_model, np_utils

INPUT_FILE = '/Users/shuchendu/Downloads/cornell movie-dialogs corpus/movie_lines.txt'

with open(INPUT_FILE, 'rb') as f:
    input_data = f.read().decode('utf-8', errors='ignore')

# actor lines
actors, lines = [], []
input_lines = input_data.split('\n')
maxLen = 0
for i, input_line in enumerate(input_lines):
    if i > 1000:
        break
    if not input_line:
        continue
    line_tokens = input_line.split('+++$+++')
    actor, line = line_tokens[3], line_tokens[4]
    actors.append(actor)
    line = text_to_word_sequence(line)
    maxLen = max(maxLen, len(line))
    lines.append(line)

# tokenize
tkn = Tokenizer()
texts = [w for sent in lines for w in sent]
tkn.fit_on_texts(texts)
word2idx = tkn.word_index
idx2word = {v: k for k, v in word2idx.items()}

# dialogue data
dialogue_X, dialogue_Y = [], []
last_actor, last_line = actors[0], lines[0]
for actor, line in zip(actors, lines):
    if actor == last_actor:
        last_line = line
        continue
    else:
        dialogue_X.append([word2idx[w] for w in last_line])
        dialogue_Y.append([word2idx[w] for w in line])
        last_actor, last_line = actor, line
dialogue_X = pad_sequences(dialogue_X, maxlen=maxLen)
dialogue_Y = pad_sequences(dialogue_Y, maxlen=maxLen)
#dialogue_Y = np_utils.to_categorical(dialogue_Y, num_classes=len(word2idx))

# split train/test data
len_train = int(0.8 * len(dialogue_X))
dialogue_X_train, dialogue_X_test = dialogue_X[: len_train], dialogue_X[len_train:]
dialogue_Y_train, dialogue_Y_test = dialogue_Y[: len_train], dialogue_Y[len_train:]

# model definition
EMBEDDING_SIZE = 1000
HIDDEN_SIZE = 1000
DENSE_SIZE = 1000

dialogue_input = Input(shape=(maxLen,))
embedding = Embedding(input_dim=len(word2idx) + 1,
                      output_dim=EMBEDDING_SIZE,
                      embeddings_initializer='glorot_uniform',
                      input_length=maxLen)(dialogue_input)

encode1 = LSTM(HIDDEN_SIZE, return_sequences=True)(embedding)
encode2 = LSTM(HIDDEN_SIZE)(encode1)

encode2 = RepeatVector(maxLen)(encode2)

decode1 = LSTM(HIDDEN_SIZE, return_sequences=True)(encode2)
decode2 = LSTM(HIDDEN_SIZE, return_sequences=True)(decode1)

dialogue_output = TimeDistributed(Dense(DENSE_SIZE, activation='softmax'))(decode2)

model = Model(inputs=dialogue_input, outputs=dialogue_output)
plot_model(model, 'seq2seq.png', show_shapes=True)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(dialogue_X_train, dia)