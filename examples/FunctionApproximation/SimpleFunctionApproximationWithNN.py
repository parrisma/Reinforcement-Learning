import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

seed = 42
size = 200
max_x = float(2.5 * (2 * math.pi))


## Define our weird function for this excercise
def weird_function(x):
    """
    Returns the y value for the given x using the following formula
    f(x)=0.2+0.4x^2+0.3xsin(15x)+0.05cos(50x)
    """
    # y = 0.2 + 0.4 * math.pow(x, 2) + 0.3 * x * math.sin(15 * x) + 0.05 * math.cos(50 * x)
    y = math.sin(x) * math.exp(1 - math.fabs((x - (max_x / 2.0)) / (max_x / 2.0)))

    return y


## Plot our weird function
x = np.linspace(0., max_x, size)
df = pd.DataFrame({'x_definition': x})
df['y_definition'] = df['x_definition'].apply(lambda x: weird_function(x))

# g = sns.FacetGrid(df, size=4, aspect=1.5)
# g.map(plt.plot, "x_definition", "y_definition")
# plt.show()
# print("")

## Set the mean, standard deviation, and size of the dataset, respectively
mu, sigma = 0, 0.01

## Create a uniformally distributed set of X values between 0 and 1 and store in pandas dataframe
x = np.random.uniform(0, max_x, size)
df['x'] = x

## Find the "perfect" y value corresponding to each x value given
df['y_perfect'] = df['x'].apply(lambda x: weird_function(x))

## Create some noise and add it to each "perfect" y value to create a realistic y dataset
df['noise'] = np.random.normal(mu, sigma, size=(size,))
df['y'] = df['y_perfect'] + df['noise']


## Create our model with a single dense layer, with a linear activation function and glorot (Xavier) input normalization

def baseline_model_2():
    raise NotImplementedError
    # create model
    model = Sequential()
    model.add(Dense(2000, activation='relu', input_dim=1, kernel_initializer='uniform'))
    model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(units=1, activation='linear'))
    optim = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optim)
    model.summary()
    return model

def baseline_model():
    model = Sequential()
    model.add(Dense(2000, input_dim=1, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(250, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.001)
                  )
    return model


md = baseline_model()
history = md.fit(x=df['x'], y=df['y'], validation_split=0.2, batch_size=32, epochs=800, verbose=2)

## Create our predicted y's based on the model
df['y_predicted'] = md.predict(df['x'], batch_size=32, verbose=0)

g = sns.FacetGrid(df, height=4, aspect=1.5)
g.map(plt.plot, "x_definition", "y_definition")
g.map(plt.scatter, "x", "y_perfect", color='blue')
g.map(plt.scatter, "x", "y", color='green')
g.map(plt.scatter, "x", "y_predicted", color='red')
plt.show()
print("")
