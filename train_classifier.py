import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)

    padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue  # empty list/array
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        elif truncating == 'post':
            trunc = seq[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[0] == 0:
            continue

        if padding == 'post':
            padded_sequences[i, :len(trunc)] = trunc
        elif padding == 'pre':
            padded_sequences[i, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')

    return padded_sequences

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Pad sequences to the same length
data = pad_sequences(data, maxlen=42)  # Assuming each hand has 21 landmarks, x and y coordinates

# Duplicate data to match the required 84 features (if needed)
data = np.hstack((data, data))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

