import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from load_base import Load_dados
from sklearn.model_selection import cross_val_score
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')

load = Load_dados()
x_trein, y_trein, x_val, y_val = load.load_aac()

class Individuo:

    def __init__(self):
        self.cromossomo = [] #Lista com todos os genes do individuo

        totalcamandas = random.randint(1, 2) #Total de camadas ocultas
        camadas_neu = random.sample([100,200,300], totalcamandas) #quantidade de neurônios em cada camada
        activation =  random.choice(['identity', 'logistic', 'tanh', 'relu']) #Função de ativação do individuo
        solver = random.choice(['lbfgs','sgd','adam'])
        taxa_aprendizagem = random.sample([0.001, 0.01, 0.1],1)#'learning_rate_init': [0.001, 0.01, 0.1],

        self.cromossomo.append(camadas_neu) #Adiciona as camadas com neurônios no cromossomo
        self.cromossomo.append(activation)
        self.cromossomo.append(solver)
        self.cromossomo.append(taxa_aprendizagem[0])
        self.fitness = self.fitness(self.cromossomo)

    def fitness(self,crom):
        global x_trein, y_trein, x_val, y_val
        modelo = MLPClassifier(hidden_layer_sizes=crom[0], activation=crom[1],solver=crom[2],learning_rate_init=crom[3])
        scores = cross_val_score(modelo, x_trein, y_trein, cv=5, scoring='accuracy')
        accuracy = np.mean(scores)
        #modelo.fit(x_trein,y_trein)
        #predict = modelo.predict(x_val)
        #accuracy2 = accuracy_score(y_val, predict)
        #print(modelo.get_params())
        #total = accuracy + accuracy2
        return accuracy*100

class AG:

    def __init__(self, tamPop):
        self.tamPop = tamPop
        self.populacao = []

    def iniciarPop(self):
        for i in range(self.tamPop):
            individuo = Individuo() #Gerar um novo indivíduo
            self.populacao.append(individuo) #Adicionar individuo gerado na população

    # seleão dos individuos para o cruzamento
    def torneio_simples(self):
        tuor = 2
        posicaoRetorno = -1
        cont = 0
        while cont < tuor:
            selecionado = random.randint(0, len(self.populacao) - 1)
            #print(cont," Entrou ",self.populacao[selecionado].fitness)
            if (self.populacao[selecionado].fitness >= self.populacao[posicaoRetorno].fitness):
                #print("Trocou")
                posicaoRetorno = selecionado
            cont += 1
        #print("Maior ",self.populacao[posicaoRetorno].fitness)
        return posicaoRetorno

    def crossover_1ponto(self,pai1,pai2):
        filho1 = copy.deepcopy(pai1)
        filho2 = copy.deepcopy(pai2)

        #print(filho1.cromossomo)
        ponto_corte = random.randint(0,len(pai1.cromossomo)-1)
        #print(ponto_corte)

        filho1.cromossomo[0:ponto_corte] = pai1.cromossomo[0:ponto_corte]
        filho1.cromossomo[ponto_corte:len(pai1.cromossomo)] = pai2.cromossomo[ponto_corte:len(pai1.cromossomo)]

        filho2.cromossomo[0:ponto_corte] = pai2.cromossomo[0:ponto_corte]
        filho2.cromossomo[ponto_corte:len(pai2.cromossomo)] = pai1.cromossomo[ponto_corte:len(pai1.cromossomo)]

        #print("Pai 1 ",pai1.cromossomo,"  ",pai1.fitness)
        #print("Pai 2 ",pai2.cromossomo,"  ",pai2.fitness)
        #print("Filho 1 ",filho1.cromossomo)
        #print("Filho 2 ", filho2.cromossomo)
        filho1.fitness = Individuo.fitness(self,filho1.cromossomo)
        filho2.fitness = Individuo.fitness(self,filho2.cromossomo)
        return filho1,filho2

    def ordenar_maior(self):
        self.populacao = sorted(self.populacao, key=lambda populacao: populacao.fitness, reverse=True)

    def mutacao(self,cromossomo):
        #print(cromossomo)
        cromossomo_mutado = copy.deepcopy(cromossomo)
        #mudar para altomáico depois
        posicao = random.randint(0, 3)
        if posicao == 0:
            totalcamandas = random.randint(1, 2)  # Total de camadas ocultas
            camadas_neu = random.sample([100, 200, 300], totalcamandas)  # quantidade de neurônios em cada camada
            cromossomo_mutado[0] = camadas_neu
        elif posicao == 1:
            activation = random.choice(['identity', 'logistic', 'tanh', 'relu'])  # Função de ativação do individuo
            cromossomo_mutado[1] = activation
        elif posicao == 2:
            solver = random.choice(['lbfgs', 'sgd', 'adam'])
            cromossomo_mutado[2] = solver
        else:
            taxa_aprendizagem = random.sample([0.001, 0.01, 0.1],1)#'learning_rate_init': [0.001, 0.01, 0.1],
            cromossomo_mutado[3] = taxa_aprendizagem[0]
        #print(cromossomo_mutado)
        return cromossomo_mutado

    def geracao(self):
        auxiliar = copy.deepcopy(self.populacao)
        filhos = []
        taxamutacao = int((30 * len(self.populacao) / 100))
        taxacrossover = int((70 * len(self.populacao) / 100)/2)
        for i in range(taxacrossover):
            #print("Crossover")
            p1 = self.torneio_simples()
            pai1 = self.populacao[p1]
            self.populacao.remove(pai1)

            p2 = self.torneio_simples()
            pai2 = self.populacao[p2]
            self.populacao.remove(pai2)
            filhos.extend(self.crossover_1ponto(pai1,pai2))

        #print("Total filhos ",len(filhos))
        self.populacao.clear()
        self.populacao.extend(filhos)
        self.ordenar_maior()
        filhos.clear()
        filhos.extend(self.populacao)
        melhor = filhos[0]
        for j in range(taxamutacao):
            #print("Mutação")
            posicao = random.randint(1, len(filhos)-1)
            cromossomo_mutado = self.mutacao(filhos[posicao].cromossomo)
            #print(cromossomo_mutado)
            fitness_mutado = Individuo.fitness(self,cromossomo_mutado)
            #print("Fitness mutado ",fitness_mutado)
            filhos[posicao].fitness = fitness_mutado

        self.populacao.clear()
        self.populacao.append(melhor)
        self.populacao.extend(filhos)
        self.populacao.extend(auxiliar)
        self.ordenar_maior()
        self.populacao = self.populacao[0:self.tamPop]
        #print("Nova população")
        #for i in self.populacao:
            #print(i.cromossomo)
            #print(i.fitness)




if __name__ == '__main__':
    total_geracao = 30
    ag = AG(30)
    ag.iniciarPop()
    #print("População inicial")
    ag.ordenar_maior()
    '''
    for i in ag.populacao:
        print(i.cromossomo)
        print(i.fitness)
    '''
    geracao = 0
    while geracao < total_geracao:
        print(ag.populacao[0].cromossomo)
        print(ag.populacao[0].fitness)
        ag.geracao()
        geracao+=1