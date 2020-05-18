import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score


class MLP(BaseEstimator, ClassifierMixin):

    def __init__(self, layers_size=None, batch_size=None, learning_rate=0.01, epochs=300,
                 lambd=0.1, beta1=0.9, beta2=0.999, verbose=True):
        self.weights = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers_size = layers_size
        self.lambd = lambd
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose

    def sigmoid(self, Z):
        A = np.divide(1., 1 + np.exp(-Z))
        return A

    def softsign(self, Z):
        A = np.divide(Z, (1 + np.abs(Z)))
        return A

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def leaky_relu(self, Z):
        A = np.maximum(0.01 * Z, Z)
        return A

    def get_inital_weights(self):
        np.random.seed(1)
        weights = []
        for index in range(1, len(self.layers_size)):
            weights.append([(np.random.randn(self.layers_size[index], self.layers_size[index - 1]) * (
                                2 / np.sqrt(self.layers_size[index - 1]))),
                               np.zeros((self.layers_size[index], 1))])
        return weights

    def get_mini_batches(self, X, Y):
        quantidade_batchs = X.shape[1] // self.batch_size
        mini_batches = []
        batch = 0
        for batch in range(quantidade_batchs - 1):
            x = X[:, self.batch_size * batch:self.batch_size * self.batch_size * (batch + 1)]
            y = Y[:, self.batch_size * batch:self.batch_size * self.batch_size * (batch + 1)]
            mini_batches.append((x, y))

        x = X[:, self.batch_size * (batch + 1):]
        y = Y[:, self.batch_size * (batch + 1):]
        mini_batches.append((x, y))
        return mini_batches

    def init_adam(self, parametros):
        gradientes_exp_ponderados = []
        for w, b in parametros:
            w = np.zeros(w.shape)
            b = np.zeros(b.shape)
            gradientes_exp_ponderados.append([w, b])
        return gradientes_exp_ponderados, np.copy(gradientes_exp_ponderados)

    def fit(self, X, y):

        weights = self.get_inital_weights()
        X = X.to_numpy().T
        y = y.to_numpy().reshape((1, X.shape[1]))

        mini_batches = self.get_mini_batches(X, y)
        momentum, rms_prop = self.init_adam(weights)
        costs = []
        lower_cost = 9999999
        weights_with_lower_cost = None

        for epoch in range(1, self.epochs):
            for x, y in mini_batches:
                A, cache = self.forward_propagation(x, weights)

                #if 0 in A or 1 in A:
                #    print("aqui")
                A[A == 0] = 0.00000000001
                A[A == 1] = 0.99999999999
                cost = self.get_cost(A, y, weights)
                if cost < lower_cost:
                    lower_cost = cost
                    weights_with_lower_cost = np.copy(weights)
                costs.append(cost)
                grads = self.backward_propagation(A, cache, y)
                weights, momentum, rms_prop = self.get_update_weights(weights, grads, x.shape[1],
                                                                      momentum, rms_prop, epoch)
            if epoch % 100 == 0 and self.verbose:
                    print("Iteração: {} | Custo: {}".format(epoch, cost))

        self.weights = weights_with_lower_cost
        self.costs = lower_cost

    def forward_propagation(self, X, parametros):
        cache = []
        A = X
        layers_quantity = len(parametros)
        for index_layer in range(0, layers_quantity - 1):
            W, b = parametros[index_layer]
            Z = np.dot(W, A) + b
            cache.append((A, Z, W, b))
            A = self.relu(Z)
        W, b = parametros[index_layer + 1]
        Z = np.dot(W, A) + b
        cache.append((A, Z, W, b))
        A = self.sigmoid(Z)

        return A, cache

    def get_cost(self, A, Y, weights):
        m = Y.shape[1]
        l2 = (1 / m) * (self.lambd / 2) * np.sum([np.sum(np.square(w)) for w in weights[:][0]])
        cost = (1. / m) * (-np.dot(Y, np.log(A).T) - np.dot(1 - Y, np.log(1 - A).T)) + l2
        cost = float(np.squeeze(cost))
        return cost

    def derivative_relu(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        # dZ[Z > 0] = 1
        return dZ

    def derivative_sigmoid(self, dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    def backward_propagation(self, A, cache, Y):
        m = A.shape[1]
        Y = Y.reshape(A.shape)
        grads = []

        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        A_prev, Z, W, b = cache[-1]
        dZ = self.derivative_sigmoid(dA, Z)

        dA_prev_layer = np.dot(W.T, dZ)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        grads.append([dW, db])

        for cache_ in cache[::-1][1:]:
            dA = dA_prev_layer
            A_prev, Z, W, b = cache_
            dZ = self.derivative_relu(dA, Z)

            dA_prev_layer = np.dot(W.T, dZ)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            grads.append([dW, db])

        return grads[::-1]

    def get_update_weights(self, weights, gradientes, m, momentum, rms_prop, iteracao):
        L = len(weights)
        for l in range(L):
            dW = (gradientes[l][0] + (self.lambd / m) * weights[l][0])
            db = gradientes[l][1]
            momentum[l][0] = self.beta1 * momentum[l][0] + (1 - self.beta1) * dW
            momentum[l][1] = self.beta1 * momentum[l][1] + (1 - self.beta1) * db
            momentum_bias_corregido_w = momentum[l][0] / (1 - self.beta1 ** iteracao)
            momentum_bias_corregido_b = momentum[l][1] / (1 - self.beta1 ** iteracao)

            rms_prop[l][0] = self.beta2 * rms_prop[l][0] + (1 - self.beta2) * np.square(dW)
            rms_prop[l][1] = self.beta2 * rms_prop[l][1] + (1 - self.beta2) * np.square(db)
            rms_prop_bias_corregido_w = rms_prop[l][0] / (1 - self.beta2 ** iteracao)
            rms_prop_bias_corregido_b = rms_prop[l][1] / (1 - self.beta2 ** iteracao)

            weights[l][0] -= self.learning_rate * np.divide(momentum_bias_corregido_w,
                                                            np.sqrt(rms_prop_bias_corregido_w) + 0.0000000001)
            weights[l][1] -= self.learning_rate * np.divide(momentum_bias_corregido_b,
                                                            np.sqrt(rms_prop_bias_corregido_b) + 0.0000000001)

        return weights, momentum, rms_prop

    def predict(self, X):
        X = X.to_numpy().T
        A, cache = self.forward_propagation(X, self.weights)
        return np.round(A)

    def score(self, X, y, sample_weight=None):
        #X = X.to_numpy().T
        y = y.to_numpy().reshape((1, X.shape[0]))
        y_pred = self.predict(X)
        return f1_score(y[0], y_pred[0])



if __name__ == "__main__":
    df = pd.read_csv("cardio_train.csv", delimiter=";", index_col=0)
    mmsc = MinMaxScaler()
    variaveis_continuas = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol"]
    df[variaveis_continuas] = mmsc.fit(df[variaveis_continuas]).transform(df[variaveis_continuas])
    df.gender = df.gender.apply(lambda genero: 0 if genero == 2 else genero)
    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = MLP(beta1=0.83, epochs=300, lambd=0.2, batch_size=4096, layers_size=[11, 25, 5, 3, 1],
              learning_rate=0.03)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    print(clf)
    #X_train = X_train.T
    #X_test = X_test.T
    #y_train = y_train.reshape((1, X_train.shape[1]))
    #y_test = y_test.reshape((1, X_test.shape[1]))
    # param_grid = dict(tamanho_layers=[[11, 8, 4, 2, 1], [11, 10, 8, 6, 4, 2, 1], [11, 8, 8, 4, 4, 2, 1],
    #                                   [11, 8, 7, 6, 5, 3, 1]],
    #                   tamanho_batch=[4096],
    #                   taxa_aprendizado=[0.01, 0.03, 0.05],
    #                   iteracoes=[200],
    #                   lambd=[0.2, 0.4, 0.6],
    #                   beta1=np.asarray(range(80, 91, 4)) * 0.01,
    #                   verbose=[False]
    #                   )
    #clf = MLP(beta1=0.8, beta2=0.999, iteracoes=200, lambd=200, tamanho_batch=4096,
    #          tamanho_layers=[11,8,7,6,5,3,1], taxa_aprendizado=0.05)
    #clf.fit(X_train, y_train)
    #print(clf)
    #clf = GridSearchCV(MLP(), param_grid, n_jobs=3, verbose=10)
    #clf.fit(X_train, y_train)
    #print(clf)


    #classificador = MLP(tamanho_layers=[11, 8, 7, 6, 5, 4, 2, 1], tamanho_batch=4096)
    #classificador.fit(X_train, y_train)
    #print(classificador)
