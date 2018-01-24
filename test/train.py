from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from seq2seq import seq2seq, attention_seq2seq
from collections import defaultdict
from itertools import izip

INPUT_FILE = 'data/movie_lines.txt'

with open(INPUT_FILE, 'rb') as f:
    input_data = f.read().decode('utf-8', errors='ignore')

# actor lines
actors, lines = [], []
input_lines = input_data.split('\n')
# update max sentence length while not exceeding 250
maxLen = 0
for i, input_line in enumerate(input_lines):
    if i > 2000: break
    if not input_line:
        continue
    line_tokens = input_line.split('+++$+++')
    actor, line = line_tokens[3], line_tokens[4]
    actors.append(actor)
    line = text_to_word_sequence(line)
    maxLen = min(max(maxLen, len(line)), 250)
    lines.append(line)

# tokenize
tkn = Tokenizer()
texts = [w for sent in lines for w in sent]
tkn.fit_on_texts(texts)
word2idx = tkn.word_index
idx2word = {v: k for k, v in word2idx.items()}

# take the most frequent 2000 words if the original vocab is too large
vocab_size = min(2000, len(idx2word)) + 2
idx2vocab = {}
idx2vocab[0] = u'PAD'
idx2vocab[1] = u'UNK'
for i in range(2, vocab_size):
    idx2vocab[i] = idx2word[i-1]

# use default dict to map unknown words to index 1
vocab2idx = {v: k for k, v in idx2vocab.items()}
default_vocab2idx = defaultdict(lambda: 1, vocab2idx)

# dialogue data
dialogue_X, dialogue_Y = [], []
last_actor, last_line = actors[0], lines[0]
for actor, line in izip(actors, lines):
    if actor == last_actor:
        last_line = line
        continue
    else:
        dialogue_X.append([default_vocab2idx[w] for w in last_line])
        dialogue_Y.append([default_vocab2idx[w] for w in line])
        last_actor, last_line = actor, line
dialogue_X = pad_sequences(dialogue_X, maxlen=maxLen, padding='post', truncating='post')
dialogue_Y = pad_sequences(dialogue_Y, maxlen=maxLen, padding='post', truncating='post')
dialogue_Y = np_utils.to_categorical(dialogue_Y, num_classes=vocab_size)

# split train/test data
len_train = int(0.8 * len(dialogue_X))
dialogue_X_train, dialogue_X_test = dialogue_X[: len_train], dialogue_X[len_train:]
dialogue_Y_train, dialogue_Y_test = dialogue_Y[: len_train], dialogue_Y[len_train:]

# create models
model = seq2seq(maxLen, vocab_size)
#model = attention_seq2seq(maxLen, vocab_size)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(dialogue_X_train, dialogue_Y_train, batch_size=256, epochs=50, verbose=1)
model.save('./seq2seq.h5')
