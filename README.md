# Rede Neural Artificial do Zero – Conjunto de Dados Iris

Trabalho da disciplina de **Inteligência Artificial**, do curso de **Ciência da Computação** do Centro Universitário Presidente Antônio Carlos (UNIPAC Barbacena).

| | |
|---|---|
| **Aluno** | João Emanuel de Faria |
| **Matrícula** | 241-007675 |
| **Disciplina** | Inteligência Artificial |
| **Professor** | Felipe Roncalli de Paula Carneiro |

## Sobre o projeto

Este projeto implementa e treina, do zero, uma Rede Neural Artificial com uma camada escondida para classificar as três espécies de flores do conjunto de dados **Iris** (*setosa*, *versicolor* e *virginica*) a partir de 4 características: comprimento e largura da sépala e da pétala.

A rede foi construída usando apenas **NumPy** para as operações matriciais. O **Scikit-Learn** foi utilizado estritamente para carregar e preparar os dados.

## Arquitetura da rede

| Camada | Neurônios | Ativação |
|---|---|---|
| Entrada | 4 | – |
| Escondida | 8 | ReLU |
| Saída | 3 | Softmax |

- **Função de perda:** Cross-Entropy Loss (entropia cruzada)
- **Otimização:** Gradiente Descendente
- **Taxa de aprendizado:** 0.1
- **Épocas:** 1000
- **Inicialização:** pesos (W1, W2) aleatórios e vieses (b1, b2) zerados

## Etapas implementadas

1. **Pré-processamento:** carregamento do Iris com `load_iris`, One-Hot Encoding dos rótulos, divisão em treino (80%) e teste (20%) e padronização das features (média 0 e desvio padrão 1, ajustada apenas no conjunto de treino).
2. **Forward Pass:** cálculo das ativações da camada escondida (ReLU) e da camada de saída (Softmax).
3. **Cálculo da perda:** Cross-Entropy Loss implementada manualmente.
4. **Backpropagation:** gradientes de W1, b1, W2 e b2 calculados com a regra da cadeia.
5. **Atualização dos pesos:** subtração da taxa de aprendizado multiplicada pelos gradientes.
6. **Avaliação:** inferência no conjunto de teste e cálculo da acurácia final.

## Como executar

**1. Clone o repositório**

```bash
git clone https://github.com/Joao-EF/Rede_Neural_Artificial-Conjunto_Iris.git
cd Rede_Neural_Artificial-Conjunto_Iris
```

**2. Crie e ative um ambiente virtual (opcional, recomendado)**

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

**3. Instale as dependências**

```bash
pip install -r requirements.txt
```

**4. Execute o script**

```bash
python rede_neural.py
```

## Resultado

Durante o treinamento, o programa imprime a perda a cada 100 épocas. Ao final, exibe a acurácia no conjunto de teste.

**Acurácia final no conjunto de teste:** 96,67%

## Estrutura do repositório

```
├── rede_neural.py     # implementação da rede neural e treinamento
├── requirements.txt   # dependências do projeto
├── README.md          # documentação
└── .gitignore         # arquivos ignorados pelo Git
```

## Tecnologias

- Python 3
- NumPy
- Scikit-Learn (somente para carregar e preparar os dados)
