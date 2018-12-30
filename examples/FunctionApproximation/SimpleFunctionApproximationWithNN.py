import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

seed = 42


## Define our weird function for this excercise
def weird_function(x):
    """
    Returns the y value for the given x using the following formula
    f(x)=0.2+0.4x^2+0.3xsin(15x)+0.05cos(50x)
    """
    y = 0.2 + 0.4 * math.pow(x, 2) + 0.3 * x * math.sin(15 * x) + 0.05 * math.cos(50 * x)

    return y


size = 200
## Plot our weird function
x = np.linspace(0., 1., size)
df = pd.DataFrame({'x_definition': x})
df['y_definition'] = df['x_definition'].apply(lambda x: weird_function(x))

# g = sns.FacetGrid(df, size=4, aspect=1.5)
# g.map(plt.plot, "x_definition", "y_definition")
# plt.show()
# print("")

## Set the mean, standard deviation, and size of the dataset, respectively
mu, sigma = 0, 0.01

## Create a uniformally distributed set of X values between 0 and 1 and store in pandas dataframe
x = np.random.uniform(0, 1, size)
df['x'] = x

## Find the "perfect" y value corresponding to each x value given
df['y_perfect'] = df['x'].apply(lambda x: weird_function(x))

## Create some noise and add it to each "perfect" y value to create a realistic y dataset
df['noise'] = np.random.normal(mu, sigma, size=(size,))
df['y'] = df['y_perfect'] + df['noise']


# g = sns.FacetGrid(df, size=4, aspect=1.5)
# g.map(plt.plot, "x_definition", "y_definition")
# g.map(plt.scatter, "x", "y_perfect", color='blue')
# g.map(plt.scatter, "x", "y", color='green')

## Create our model with a single dense layer, with a linear activation function and glorot (Xavier) input normalization

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(2000, activation='relu', input_dim=1, init='uniform'))
    # model.add(Dropout(0.25))
    model.add(Dense(500, activation='relu', init='uniform'))
    # model.add(Dropout(0.25))
    model.add(Dense(500, activation='relu', init='uniform'))
    model.add(Dense(500, activation='relu', init='uniform'))
    model.add(Dense(500, activation='relu', init='uniform'))
    # model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu', init='uniform'))
    # model.add(Dropout(0.25))
    model.add(Dense(output_dim=1, activation='linear'))
    optim = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optim)
    return model


md = baseline_model()
history = md.fit(x=df['x'], y=df['y'], validation_split=0.2, batch_size=32, nb_epoch=800, verbose=2)

## Create our predicted y's based on the model
df['y_predicted'] = md.predict(df['x'], batch_size=32, verbose=0)

g = sns.FacetGrid(df, size=4, aspect=1.5)
g.map(plt.plot, "x_definition", "y_definition")
g.map(plt.scatter, "x", "y_perfect", color='blue')
g.map(plt.scatter, "x", "y", color='green')
g.map(plt.scatter, "x", "y_predicted", color='red')
plt.show()
print("")
