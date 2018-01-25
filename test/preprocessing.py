from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from seq2seq import seq2seq, attention_seq2seq
from keras.utils import np_utils
from keras.models import load_model
from collections import defaultdict
from itertools import izip
import numpy as np
import os

INPUT_FILE = '../data/movie_lines.txt'

# read raw data
with open(INPUT_FILE, 'rb') as f:
	input_data = f.read().decode('utf-8', errors='ignore')

# actor lines
actors, lines = [], []
input_lines = input_data.split('\n')
# update max sentence length while not exceeding 250
maxLen = 0
for i, input_line in enumerate(input_lines):
	if i > 5000: break
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
print('Total vocab is {}'.format(len(idx2word)))

# take the most frequent 2000 words if the original vocab is too large
vocab_size = min(2000, len(idx2word)) + 2
idx2vocab = {}
idx2vocab[0] = u'PAD'
idx2vocab[1] = u'UNK'
for i in xrange(2, vocab_size):
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
train_n_samples, test_n_samples = len(dialogue_X_train), len(dialogue_X_test)
print('Shape of training X is {}, Y is {}'.format(dialogue_X_train.shape, dialogue_Y_train.shape))

# save train/test data
def save_data(fpath, X_data, Y_data, update=False):
	if os.path.isfile(fpath) and not update:
		print('{} already exists...'.format(fpath))
	else:
		np.savez_compressed(fpath, X=X_data, Y=Y_data)
train_fpath, test_fpath = '../data/train.npz', '../data/test.npz'
save_data(train_fpath, dialogue_X_train, dialogue_Y_train)
save_data(test_fpath, dialogue_X_test, dialogue_Y_test)

# generate train/test data
def generate_data(fpath, batch_size, n_samples):
	data = np.load(fpath)	
	dialogue_X, dialogue_Y = data['X'], data['Y']
        while True:
	    for i in xrange(0, n_samples, batch_size):
	        X, Y = dialogue_X[i: i + batch_size], dialogue_Y[i: i + batch_size]
                yield (X, Y)
        data.close()

# train if model doesn't exist
BATCH_SIZE = 64
if os.path.isfile('./seq2seq.h5'):
	print('model already exists...')
else:
	# create models
	model = seq2seq(maxLen, vocab_size)
	#model = attention_seq2seq(maxLen, vocab_size)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# train models
	model.fit_generator(generate_data(train_fpath, BATCH_SIZE, train_n_samples), 
				steps_per_epoch=train_n_samples/BATCH_SIZE, epochs=50)

	# save models
	model.save('./seq2seq.h5')
	print('model saved...')

# load models
model = load_model('seq2seq.h5')
#model.summary()
print('model loaded...')

# predict
predict_result = model.predict_generator(generate_data(test_fpath, BATCH_SIZE, test_n_samples), 
						steps=test_n_samples/BATCH_SIZE, verbose=1)
print('finished predicting...')

# compare test and predict
test = np.load(test_fpath)['Y']
print('shape of test and prediction: {}, {}'.format(test.shape, predict_result.shape))
for i in xrange(1):
	_test, _pred = test[i], predict_result[i]
	_test_line, _pred_line = [], []
	for row_test, row_pred in izip(_test, _pred):
		_test_line.append(idx2vocab[np.argmax(row_test)])
		_pred_line.append(idx2vocab[np.argmax(row_pred)])
		print(row_pred)
	_test_line, _pred_line = ' '.join(_test_line), ' '.join(_pred_line)
	#print('No. {}: [test] {}, [pred] {}'.format(i, _test_line, _pred_line))
	#print('No. {}: [test line] {}'.format(i, np.argmax(_test, axis=1)))
	#print('No. {}: [pred line] {}'.format(i, np.argmax(_pred, axis=1)))
