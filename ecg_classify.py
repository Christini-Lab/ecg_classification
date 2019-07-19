import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import itertools
import collections
from tensorflow.keras import datasets, layers, models, utils

plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 

# Train Inputs
def get_train_inputs():
    x = tf.constant(X_train)
    y = tf.constant(y_train)
    return x, y

# Test Inputs
def get_test_inputs():
    x = tf.constant(X_test)
    y = tf.constant(y_test)
    return x, y

# Eval data
def get_eval_data():
    return tf.constant(X_test)

# Plot matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #cm[i, j] = 0 if np.isnan(cm[i, j]) else cm[i, j]
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Variables

path = 'mitDB/'
window_size = 160
maximum_counting = 10000

classes = ['N', 'L', 'R', 'A', 'V', '/']
n_classes = len(classes)
count_classes = [0]*n_classes

X = list()
y = list()



# Read files
filenames = next(os.walk(path))[2]

# Split and save .csv , .txt
records = list()
annotations = list()
filenames.sort()

for f in filenames:
    filename, file_extension = os.path.splitext(f)

    # *.csv
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)

    # *.txt
    else:
        annotations.append(path + filename + file_extension)

# Records
for r in range(0,len(records)):
    signals = []

    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file\
        row_index = -1
        for row in spamreader:
            if(row_index >= 0):
                signals.insert(row_index, int(row[1]))
            row_index += 1

    # Read anotations: R position and Arrhythmia class
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines()
        beat = list()

        for d in range(1, len(data)): # 0 index is Chart Head
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted) # Time... Clipping
            pos = int(next(splitted)) # Sample ID
            arrhythmia_type = next(splitted) # Type

            if(arrhythmia_type in classes):
                arrhythmia_index = classes.index(arrhythmia_type)
                if count_classes[arrhythmia_index] > maximum_counting: # avoid overfitting
                    pass
                else:
                    count_classes[arrhythmia_index] += 1
                    if(window_size < pos and pos < (len(signals) - window_size)):
                        beat = signals[pos-window_size+1:pos+window_size]
                        X.append(beat)
                        y.append(arrhythmia_index)

# np.shape(X) => (42021, 319)
# np.shape(y) => (42021, )

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print("X_train : ", len(X_train))
print("X_test  : ", len(X_test))
print("y_train : ", collections.Counter(y_train))
print("y_test  : ", collections.Counter(y_test))

print(np.shape(X_train[0]))
print(np.shape(y_train))
print(np.shape(X_test))
print(np.shape(y_test))

print(y_train[0])

# Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = normalize(X_train, axis=0, norm='max')
X_test = normalize(X_test, axis=0, norm='max')

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("ecg", shape=np.shape(X_train[0]))]

model = models.Sequential()
model.add(layers.Reshape((319, 1), input_shape=(319,)))
model.add(layers.Conv1D(10, 20, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Dropout(.2))
#model.add(layers.BatchNormalization())
#model.add(layers.Conv1D(20,10,activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Dropout(.2))
#model.add(layers.BatchNormalization())
#model.add(layers.Conv1D(20,10,activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Dropout(.2))
#model.add(layers.BatchNormalization())
model.add(layers.Conv1D(20,10,activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(6, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

y_binary = utils.to_categorical(y_train)

BATCH_SIZE = 400
EPOCHS = 4 
history = model.fit(X_train,y_binary, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

y_test_binary = utils.to_categorical(y_test)
test_loss, test_acc = model.evaluate(X_test, y_test_binary)
y_pred = model.predict(X_test)

print(test_loss)
print(test_acc)
matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
plot_confusion_matrix(matrix, classes, normalize=True)

fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')

plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.subplot(2,2,2)
plt.plot(history.history['loss'])
plt.show()
