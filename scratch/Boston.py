import numpy
import math
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

nt = 100
x = np.empty((nt, 1))
y = np.empty((nt, 1))
v = -1.0
for i in range(0, nt):
    x[i] = i
    y[i] = 0.2+0.4*math.pow(v, 2)+0.3*v*math.sin(15*v)+0.05*math.cos(50*v) # (5 * v * v * v) - (2 * v * v) + (1.2 * v) - 3
    v += 0.01

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1000, input_dim=1, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(10, input_dim=1, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, kernel_initializer='uniform', activation='linear'))
    # Compile model
    optim = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optim)
    return model

# fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

km = baseline_model()
history = km.fit(x, y, nb_epoch=500, batch_size=32, validation_split=0.2, verbose=2)
plt.scatter(x, y)
plt.scatter(x, km.predict(x, batch_size=32))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
print("")

