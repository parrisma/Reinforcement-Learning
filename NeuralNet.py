#
# Train a simple Sequential Net to learn the Q values for a given board state.
#
import keras
from keras.models import Sequential
from keras.layers import Dense
from Persistance import Persistance
from pathlib import Path
import numpy as np

# fix random seed for reproducibility
np.random.seed(42)

model_filename = "model.h5"

p = Persistance()
X, Y = p.load_as_X_Y("./qv_dump.pb")

if Path(model_filename).is_file():
    ##
    model = keras.models.load_model(model_filename)
else:
    # create model
    model = Sequential()
    model.add(Dense(1000, input_dim=10, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(9))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=50, batch_size=10)

    model.save(model_filename)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

for i in range(0, 10):
    idx = np.random.randint(0, X.shape[0])
    print("X[idx] := "+str(X[idx])+" for Y[idx]: "+str(Y[idx]))
    x = np.zeros((1,10))
    x[0] = X[idx]
    print("Predict Y:= "+str(model.predict(x)))

