# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from Persistance import Persistance
import numpy

# fix random seed for reproducibility
numpy.random.seed(42)

p = Persistance()
X, Y = p.load_as_X_Y("./qv_dump.pb")

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
# evaluate the model

model.save("/.fully_connected_1.h5")

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))