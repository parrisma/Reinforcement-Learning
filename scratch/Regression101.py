import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures

seed = 42


def baseline_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == "__main__":

    nt = 200
    x = np.empty((nt, 1))
    y = np.empty((nt, 1))
    for i in range(0, nt):
        x[i] = i
        y[i] = 2 * i * i + 1.2 * i + 3

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly_features.fit_transform(x)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    pred_poly = lin_reg.predict(x_poly)

    estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=50, shuffle=False, verbose=True)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, x_poly, y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    estimator.fit(x_poly, y, shuffle=False)
    prediction = estimator.predict(x_poly)
    plt.plot(x, prediction)
    plt.show()
