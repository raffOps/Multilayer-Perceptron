import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt


class MLP(BaseEstimator,  ClassifierMixin):

    def __init__(self, tamanho_layers=None,tamanho_batch=None, taxa_aprendizado=0.01, iteracoes=300,
                 lambd=0.1, beta1=0.9, beta2=0.999, verbose=True):
        self.parametros = None
        self.taxa_aprendizado = taxa_aprendizado
        self.iteracoes = iteracoes
        self.tamanho_layers = tamanho_layers
        self.lambd = lambd
        self.tamanho_batch = tamanho_batch
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
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

    def get_parametros_iniciais(self):
        np.random.seed(1)
        parametros = []
        for indice in range(1, len(self.tamanho_layers)):
            parametros.append([np.random.randn(self.tamanho_layers[indice], self.tamanho_layers[indice - 1]) * (
                        2 / np.sqrt(self.tamanho_layers[indice - 1])),
                               np.zeros((self.tamanho_layers[indice], 1))])
        return parametros

    def forward_propagation(self, X, parametros):
        cache = []
        A = X
        quantidade_layers = len(parametros)
        for indice_layer in range(0,
                                  quantidade_layers - 1):  # foward propagation até o ultimo layer antes do layer output
            W, b = parametros[indice_layer]
            Z = np.dot(W, A) + b
            cache.append((A, Z, W, b))
            A = self.relu(Z)
        W, b = parametros[indice_layer + 1]
        Z = np.dot(W, A) + b
        cache.append((A, Z, W, b))
        A = self.sigmoid(Z)

        return A, cache

    def get_custo(self, A, Y, parametros):
        m = Y.shape[1]
        l2 = (1 / m) * (self.lambd / 2) * np.sum([np.sum(np.square(w)) for w in parametros[:][0]])
        custo = (1. / m) * (-np.dot(Y, np.log(A).T) - np.dot(1 - Y, np.log(1 - A).T)) + l2
        custo = float(np.squeeze(custo))
        return custo

    def derivada_relu(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        # dZ[Z > 0] = 1
        return dZ

    def derivada_leaky_relu(self,  dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0.01
        # dZ[Z > 0] = 1
        return dZ

    def derivada_softsign(self, dA, Z):
        return np.multiply(dA, np.divide(1, np.multiply(1 + Z, 1 + Z)))

    def derivada_sigmoide(self, dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    def backward_propagation(self, A, cache, Y):
        m = A.shape[1]
        Y = Y.reshape(A.shape)
        gradientes = []

        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        A_prev, Z, W, b = cache[-1]
        dZ = self.derivada_sigmoide(dA, Z)

        dA_layer_anterior = np.dot(W.T, dZ)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        gradientes.append([dW, db])

        for cache_ in cache[::-1][1:]:
            dA = dA_layer_anterior
            A_prev, Z, W, b = cache_
            dZ = self.derivada_relu(dA, Z)

            dA_layer_anterior = np.dot(W.T, dZ)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            gradientes.append([dW, db])

        return gradientes[::-1]

    def get_parametros_atualizados(self, parametros, gradientes, m, momentum, rms_prop, iteracao):
        L = len(parametros)
        for l in range(L):
            dW = (gradientes[l][0] + (self.lambd / m) * parametros[l][0])
            db = gradientes[l][1]
            momentum[l][0] = self.beta1 * momentum[l][0] + (1 - self.beta1) * dW
            momentum[l][1] = self.beta1 * momentum[l][1] + (1 - self.beta1) * db
            momentum_bias_corregido_w = momentum[l][0] / (1 - self.beta1 ** iteracao)
            momentum_bias_corregido_b = momentum[l][1] / (1 - self.beta1 ** iteracao)

            rms_prop[l][0] = self.beta2 * rms_prop[l][0] + (1 - self.beta2) * np.square(dW)
            rms_prop[l][1] = self.beta2 * rms_prop[l][1] + (1 - self.beta2) * np.square(db)
            rms_prop_bias_corregido_w = rms_prop[l][0] / (1 - self.beta2 ** iteracao)
            rms_prop_bias_corregido_b = rms_prop[l][1] / (1 - self.beta2 ** iteracao)

            parametros[l][0] -= self.taxa_aprendizado * np.divide(momentum_bias_corregido_w,
                                                                  np.sqrt(rms_prop_bias_corregido_w) + 0.0000000001)
            parametros[l][1] -= self.taxa_aprendizado * np.divide(momentum_bias_corregido_b,
                                                                  np.sqrt(rms_prop_bias_corregido_b) + 0.0000000001)

        return parametros, momentum, rms_prop

    def get_mini_batches(self, X, Y):
        quantidade_batchs = X.shape[1] // self.tamanho_batch
        mini_batches = []
        batch = 0
        for batch in range(quantidade_batchs - 1):
            x = X[:, self.tamanho_batch * batch:self.tamanho_batch * self.tamanho_batch * (batch + 1)]
            y = Y[:, self.tamanho_batch * batch:self.tamanho_batch * self.tamanho_batch * (batch + 1)]
            mini_batches.append((x, y))

        x = X[:, self.tamanho_batch * (batch + 1):]
        y = Y[:, self.tamanho_batch * (batch + 1):]
        mini_batches.append((x, y))
        return mini_batches

    def inicializa_adam(self, parametros):
        gradientes_exp_ponderados = []
        for w, b in parametros:
            w = np.zeros(w.shape)
            b = np.zeros(b.shape)
            gradientes_exp_ponderados.append([w, b])
        return gradientes_exp_ponderados, np.copy(gradientes_exp_ponderados)

    def fit(self, X, y):

        parametros = self.get_parametros_iniciais()
        X = X.to_numpy().T
        y = y.to_numpy().reshape((1, X.shape[1]))

        mini_batches = self.get_mini_batches(X, y)
        momentum, rms_prop = self.inicializa_adam(parametros)
        custos = []
        menor_custo = 1
        parametros_com_menor_custo = None

        for iteracao in range(1, self.iteracoes):
            for x, y in mini_batches:
                A, cache = self.forward_propagation(x, parametros)
                custo = self.get_custo(A, y, parametros)
                if custo < menor_custo:
                    menor_custo = custo
                    parametros_com_menor_custo = np.copy(parametros)
                custos.append(custo)
                gradientes = self.backward_propagation(A, cache, y)
                parametros, momentum, rms_prop = self.get_parametros_atualizados(parametros, gradientes, x.shape[1],
                                                                            momentum, rms_prop, iteracao)
            if iteracao % 100 == 0 and self.verbose:
                print("Iteração: {} | Custo: {}".format(iteracao, custo))

        self.parametros = parametros_com_menor_custo
        return custos

    def predict(self, X):
        X = X.to_numpy().T
        A, cache = self.forward_propagation(X, self.parametros)
        return np.round(A)

    def score(self, X, y):
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
    #X_train = X_train.T
    #X_test = X_test.T
    #y_train = y_train.reshape((1, X_train.shape[1]))
    #y_test = y_test.reshape((1, X_test.shape[1]))
    param_grid = dict(tamanho_layers=[[11, 8, 4, 2, 1]],
                      tamanho_batch=[4096],
                      taxa_aprendizado=[0.01, 0.03],
                      iteracoes=[300],
                      lambd=[0.1],
                      beta1=np.asarray(range(80, 91, 3)) * 0.01,
                      verbose=[False]
                      )
    clf = GridSearchCV(MLP(), param_grid, n_jobs=3, verbose=10)
    clf.fit(X_train, y_train)
    print(clf)


    #classificador = MLP(tamanho_layers=[11, 8, 7, 6, 5, 4, 2, 1], tamanho_batch=4096)
    #classificador.fit(X_train, y_train)
    #print(classificador)