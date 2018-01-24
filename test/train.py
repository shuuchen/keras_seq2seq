from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from seq2seq import seq2seq, attention_seq2seq
from collections import defaultdict
from itertools import izip
from preprocessing import preprocessing
import numpy as np

# preprocessing
maxLen, vocab_size, total_n_samples = preprocessing()
print('preprocessing is over...')

# create models
model = seq2seq(maxLen, vocab_size)
#model = attention_seq2seq(maxLen, vocab_size)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train models
BATCH_SIZE = 64
def generate_data():
    train_data = np.load('../data/train.npz')	
    dialogue_X_train, dialogue_Y_train = train_data['X'], train_data['Y']
    while True:
	for i in range(0, total_n_samples, BATCH_SIZE):
	    X, Y = dialogue_X_train[i: i + BATCH_SIZE], dialogue_Y_train[i: i + BATCH_SIZE]
            yield (X, Y)
    train_data.close()

model.fit_generator(generate_data(), steps_per_epoch=total_n_samples/BATCH_SIZE, epochs=10)

# save models
model.save('./seq2seq.h5')
