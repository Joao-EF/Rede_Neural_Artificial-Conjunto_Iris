import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  # deixa os resultados repetíveis

# ---------------------------------------------------------------
# 1. PRÉ-PROCESSAMENTO DOS DADOS
# ---------------------------------------------------------------
iris = load_iris()
X = iris.data      # formato (150, 4)
y = iris.target    # formato (150,)

# One-Hot Encoding
Y = np.eye(3)[y]   # formato (150, 3)

# Divisão em treino (80%) e teste (20%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=y
)

# Padronização (média 0, desvio 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # aprende a média/desvio só no treino
X_test = scaler.transform(X_test)        # aplica no teste a mesma régua

print('Treino:', X_train.shape, Y_train.shape)
print('Teste:', X_test.shape, Y_test.shape)

# ---------------------------------------------------------------
# 2. ARQUITETURA: INICIALIZAÇÃO DOS PESOS E VIESES
# ---------------------------------------------------------------
n_entrada = 4
n_escondida = 8
n_saida = 3

W1 = np.random.randn(n_entrada, n_escondida) * 0.1
b1 = np.zeros((1, n_escondida))
W2 = np.random.randn(n_escondida, n_saida) * 0.1
b2 = np.zeros((1, n_saida))


# ---------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------
def relu(Z):
    return np.maximum(0, Z)


def relu_derivada(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    Z_estavel = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_estavel)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def cross_entropy(Y, A2):
    m = Y.shape[0]
    return -np.sum(Y * np.log(A2 + 1e-8)) / m


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1     # soma ponderada da camada escondida
    A1 = relu(Z1)        # ativação ReLU
    Z2 = A1 @ W2 + b2    # soma ponderada da saída
    A2 = softmax(Z2)     # probabilidades finais
    return Z1, A1, Z2, A2


def backward(X, Y, Z1, A1, A2, W2):
    m = X.shape[0]

    dZ2 = A2 - Y
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivada(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# ---------------------------------------------------------------
# 3. TREINAMENTO VIA GRADIENTE DESCENDENTE
# ---------------------------------------------------------------
epocas = 1000
taxa_aprendizado = 0.1

for epoca in range(epocas):
    # 1. Forward
    Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

    # 2. Perda
    perda = cross_entropy(Y_train, A2)

    # 3. Backward
    dW1, db1, dW2, db2 = backward(X_train, Y_train, Z1, A1, A2, W2)

    # 4. Atualizar pesos
    W1 -= taxa_aprendizado * dW1
    b1 -= taxa_aprendizado * db1
    W2 -= taxa_aprendizado * dW2
    b2 -= taxa_aprendizado * db2

    if epoca % 100 == 0:
        print(f'Época {epoca} - Perda: {perda:.4f}')

# ---------------------------------------------------------------
# 4. AVALIAÇÃO NO CONJUNTO DE TESTE
# ---------------------------------------------------------------
_, _, _, A2_teste = forward(X_test, W1, b1, W2, b2)

predicoes = np.argmax(A2_teste, axis=1)
verdadeiras = np.argmax(Y_test, axis=1)

acuracia = np.mean(predicoes == verdadeiras)
print(f'Acurácia no teste: {acuracia * 100:.2f}%')