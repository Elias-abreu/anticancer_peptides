from evolutionary_search import EvolutionaryAlgorithmSearchCV,maximize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,recall_score,matthews_corrcoef,make_scorer,roc_auc_score
from sklearn.model_selection import cross_val_score
from load_base import Load_dados
from sklearn import svm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Ag:
    def ag_mlp(self):
        #[25,50,100,250,1000,2500,5000,7600,10000]
        s1 = [100, 250, 1000, 2500]
        s2 = [100, 250, 1000, 2500]
        s3 = [100, 250, 1000, 2500]

        pares_possiveis = []
        for n1 in s1:
            for n2 in s2:
                pares_possiveis.append((n1,n2))

        pares_possiveis3 = []
        for n1 in s1:
            for n2 in s2:
                for n3 in s3:
                    pares_possiveis3.append((n1, n2,n3))

        hiper_params={
            'hidden_layer_sizes': pares_possiveis,
            'learning_rate_init': [0.001,0.01,0.1],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate':['constant', 'invscaling', 'adaptive']
        }

        ag = EvolutionaryAlgorithmSearchCV(estimator=MLPClassifier(),
                                   params= hiper_params,
                                   scoring='accuracy',
                                   population_size=10,
                                   gene_mutation_prob=0.15,
                                   gene_crossover_prob=0.7,
                                   tournament_size=2
        )

        load = Load_dados()
        x_trein, y_trein, x_val, y_val = load.load_hibrido_AAC_DPC()
        ag.fit(x_trein, y_trein)
        melhorRede = ag.best_params_
        print(melhorRede)

    def ag_svm(self):
        hiper_params = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": np.logspace(-9, 9, num=25, base=10),
            "coef0":np.logspace(-9, 9, num=25, base=10)
        }
        ag = EvolutionaryAlgorithmSearchCV(estimator= svm.SVC(),
                                           params=hiper_params,
                                           scoring='accuracy',
                                           population_size=10,
                                           gene_mutation_prob=0.15,
                                           gene_crossover_prob=0.7,
                                           tournament_size=2
                                           )

        load = Load_dados()
        x_trein, y_trein, x_val, y_val = load.load_dipeptideo()
        #print(x_trein)
        ag.fit(x_trein, y_trein)
        melhorRede = ag.best_params_
        print(ag.best_score_)
        print(melhorRede)

if __name__ == '__main__':
    a = Ag()
    a.ag_svm()






#hidden_layer_sizes: quantas camadas e quantidade de neorônios

#Ativação: 'identidade', 'logística', 'tanh', 'relu'
#solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
#learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’, Usado somente quando solver='sgd'.

#max_iter int, default = 200  (Pensei de 50 a 500, sendo de 50 em 50)

#shuffle: True, False
'''
Sortear a quantidade de camadas
sortear os neurônios entre os possíveis

'''