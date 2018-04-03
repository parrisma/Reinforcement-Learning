import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
import TemporalDifferencePolicyPersistance

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(2000, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1500, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(output_dim=9, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def read_csv():
    raw = []
    with open('Book3.csv', 'r') as csvfile:
        ttt_reader = csv.reader(csvfile, delimiter=',')
        for row in ttt_reader:
            raw.append(row)

    x = np.zeros((len(raw), 9))
    y = np.zeros((len(raw), 9))
    i = 0

    for row in raw:
        j = 0
        for c in str(row[0]):
            x[i, j] = np.float(c)
            j += 1

        j = 0
        for n in row[1:]:
            y[i, j] = np.float(n)
            j += 1
        i += 1

    return x, y



def actn(t):
    return np.max((t == np.max(t))*(0,1,2,3,4,5,6,7,8))

x, y = read_csv()

ml = baseline_model()

history = ml.fit(x=x, y=y, validation_split=0, batch_size=32, nb_epoch=500, verbose=2)

# Create our predicted y's based on the model
yp = ml.predict(x, batch_size=32, verbose=0)

yy = np.zeros((y.shape[0],1))
yyp = np.zeros((y.shape[0],1))

i = 0
for xi in y:
    yy[i] = actn(xi)
    i += 1

i = 0
for xi in yp:
    yyp[i] = actn(xi)
    i += 1

plt.scatter(yy, yyp)
plt.show()
print("")