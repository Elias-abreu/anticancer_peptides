from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,recall_score,matthews_corrcoef,make_scorer,roc_auc_score
from sklearn.model_selection import cross_val_score
from load_base import Load_dados
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Modelos_classificacao:

    def __init__(self):
        load = Load_dados()
        self.x_trein, self.y_trein, self.x_val, self.y_val = load.load_CKSAAGP()

    def modelo_svm(self):
        print("--SVM--")
        #kernel= 'rbf', C= 5.623413251903491, gamma= 5.623413251903491
        modelo = svm.SVC(C=1)
        val_scores_sensitivity = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='recall')
        especifi = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                   scoring=make_scorer(recall_score, average='macro'))
        val_scores_accuracy = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='accuracy')
        val_scores_mcc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                         scoring=make_scorer(matthews_corrcoef))
        val_scores_roc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='roc_auc')
        # t = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, labels=0))
        # print(np.mean(t))

        print(
            "Validação Interna: sensitivity {:.2f}%, specificity {:.2f}%, acurácia: {:.2f}%, MCC {:.2f}%, AUROC {:.2f}%.".format(
                np.mean(val_scores_sensitivity), np.mean(especifi), np.mean(val_scores_accuracy)*100,
                np.mean(val_scores_mcc), np.mean(val_scores_roc)))
        modelo.fit(self.x_trein, self.y_trein)
        predict = modelo.predict(self.x_val)
        accuracy = accuracy_score(self.y_val, predict)
        sensitivity = recall_score(self.y_val, predict)
        mcc = matthews_corrcoef(self.y_val, predict)
        roc = roc_auc_score(self.y_val, predict)
        tn, fp, fn, tp = confusion_matrix(self.y_val, predict).ravel()
        specificity = tn / (tn + fp)  # make_scorer(self.y_val, predict) #tn / (tn + fp)
        print(
            "Validação Externa: sensitivity {:.2f}%, specificity {:.2f}%,  acurácia {:.2f}%, MCC {:.2f}%, AUROC  {:.2f}%".format(
                sensitivity, specificity,
                accuracy*100,
                mcc, roc))

    def modelo_mlp(self):
        print("--MLP--")
        #hidden_layer_sizes= (500, 400, 300,)
        #relu, logistic
        #'hidden_layer_sizes': (50, 100), 'learning_rate_init': 0.1, 'solver': 'adam'
        #hidden_layer_sizes = (100, 250), learning_rate_init= 0.1, solver= 'adam', tol= 1e-08
        #hidden_layer_sizes= (200,500, 1000), learning_rate_init= 0.001, solver= 'adam'
        #[200, 300], 'relu', 'adam', 0.01
        #[300, 100], 'relu', 'adam', 0.001
        #modelo = MLPClassifier(hidden_layer_sizes=(500), activation='relu',solver='lbfgs',learning_rate_init=0.01)
        modelo = MLPClassifier(activation='logistic')
        val_scores_sensitivity = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='recall')
        especifi = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, average='macro'))
        val_scores_accuracy = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='accuracy')
        val_scores_mcc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(matthews_corrcoef))
        val_scores_roc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='roc_auc')
        #t = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, labels=0))
        #print(np.mean(t))

        print("Validação Interna: sensitivity {:.2f}%, specificity {:.2f}%, acurácia: {:.2f}%, MCC {:.2f}%, AUROC {:.2f}%.".format(
            np.mean(val_scores_sensitivity),np.mean(especifi), np.mean(val_scores_accuracy)*100, np.mean(val_scores_mcc), np.mean(val_scores_roc)))
        modelo.fit(self.x_trein, self.y_trein)
        predict = modelo.predict(self.x_val)
        accuracy = accuracy_score(self.y_val, predict)
        sensitivity = recall_score(self.y_val, predict)
        mcc = matthews_corrcoef(self.y_val, predict)
        roc = roc_auc_score(self.y_val, predict)
        tn, fp, fn, tp = confusion_matrix(self.y_val, predict).ravel()
        specificity = tn / (tn + fp)#make_scorer(self.y_val, predict) #tn / (tn + fp)
        print("Validação Externa: sensitivity {:.2f}%, specificity {:.2f}%,  acurácia {:.2f}%, MCC {:.2f}%, AUROC  {:.2f}%".format(sensitivity,specificity,
                                                                                                              accuracy*100,
                                                                                                              mcc, roc))

    def modelo_knn(self):
        print("--KNN--")
        modelo =  KNeighborsClassifier(n_neighbors=10)
        val_scores_sensitivity = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='recall')
        especifi = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                   scoring=make_scorer(recall_score, average='macro'))
        val_scores_accuracy = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='accuracy')
        val_scores_mcc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                         scoring=make_scorer(matthews_corrcoef))
        val_scores_roc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='roc_auc')
        # t = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, labels=0))
        # print(np.mean(t))

        print(
            "Validação Interna: sensitivity {:.2f}%, specificity {:.2f}%, acurácia: {:.2f}%, MCC {:.2f}%, AUROC {:.2f}%.".format(
                np.mean(val_scores_sensitivity), np.mean(especifi), np.mean(val_scores_accuracy)*100,
                np.mean(val_scores_mcc), np.mean(val_scores_roc)))
        modelo.fit(self.x_trein, self.y_trein)
        predict = modelo.predict(self.x_val)
        accuracy = accuracy_score(self.y_val, predict)
        sensitivity = recall_score(self.y_val, predict)
        mcc = matthews_corrcoef(self.y_val, predict)
        roc = roc_auc_score(self.y_val, predict)
        tn, fp, fn, tp = confusion_matrix(self.y_val, predict).ravel()
        specificity = tn / (tn + fp)  # make_scorer(self.y_val, predict) #tn / (tn + fp)
        print(
            "Validação Externa: sensitivity {:.2f}%, specificity {:.2f}%,  acurácia {:.2f}%, MCC {:.2f}%, AUROC  {:.2f}%".format(
                sensitivity, specificity,
                accuracy*100,
                mcc, roc))

    def modelo_RF(self):
        print("--RF--")
        modelo =  RandomForestClassifier(n_estimators=100)
        val_scores_sensitivity = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='recall')
        especifi = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                   scoring=make_scorer(recall_score, average='macro'))
        val_scores_accuracy = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='accuracy')
        val_scores_mcc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                         scoring=make_scorer(matthews_corrcoef))
        val_scores_roc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='roc_auc')
        # t = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, labels=0))
        # print(np.mean(t))

        print(
            "Validação Interna: sensitivity {:.2f}%, specificity {:.2f}%, acurácia: {:.2f}%, MCC {:.2f}%, AUROC {:.2f}%.".format(
                np.mean(val_scores_sensitivity), np.mean(especifi), np.mean(val_scores_accuracy)*100,
                np.mean(val_scores_mcc), np.mean(val_scores_roc)))
        modelo.fit(self.x_trein, self.y_trein)
        predict = modelo.predict(self.x_val)
        accuracy = accuracy_score(self.y_val, predict)
        sensitivity = recall_score(self.y_val, predict)
        mcc = matthews_corrcoef(self.y_val, predict)
        roc = roc_auc_score(self.y_val, predict)
        tn, fp, fn, tp = confusion_matrix(self.y_val, predict).ravel()
        specificity = tn / (tn + fp)  # make_scorer(self.y_val, predict) #tn / (tn + fp)
        print(
            "Validação Externa: sensitivity {:.2f}%, specificity {:.2f}%,  acurácia {:.2f}%, MCC {:.2f}%, AUROC  {:.2f}%".format(
                sensitivity, specificity,
                accuracy*100,
                mcc, roc))

    def modelo_ETree(self):
        print("--ETree--")
        modelo = ExtraTreesClassifier(max_depth=400)
        val_scores_sensitivity = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='recall')
        especifi = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                   scoring=make_scorer(recall_score, average='macro'))
        val_scores_accuracy = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='accuracy')
        val_scores_mcc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                         scoring=make_scorer(matthews_corrcoef))
        val_scores_roc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='roc_auc')
        # t = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, labels=0))
        # print(np.mean(t))

        print(
            "Validação Interna: sensitivity {:.2f}%, specificity {:.2f}%, acurácia: {:.2f}%, MCC {:.2f}%, AUROC {:.2f}%.".format(
                np.mean(val_scores_sensitivity), np.mean(especifi), np.mean(val_scores_accuracy)*100,
                np.mean(val_scores_mcc), np.mean(val_scores_roc)))
        modelo.fit(self.x_trein, self.y_trein)
        predict = modelo.predict(self.x_val)
        accuracy = accuracy_score(self.y_val, predict)
        sensitivity = recall_score(self.y_val, predict)
        mcc = matthews_corrcoef(self.y_val, predict)
        roc = roc_auc_score(self.y_val, predict)
        tn, fp, fn, tp = confusion_matrix(self.y_val, predict).ravel()
        specificity = tn / (tn + fp)  # make_scorer(self.y_val, predict) #tn / (tn + fp)
        print(
            "Validação Externa: sensitivity {:.2f}%, specificity {:.2f}%,  acurácia {:.2f}%, MCC {:.2f}%, AUROC  {:.2f}%".format(
                sensitivity, specificity,
                accuracy*100,
                mcc, roc))

    def modelo_Ridge(self):
        print("--Ridge--")
        modelo = RidgeClassifier(alpha=0)
        val_scores_sensitivity = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='recall')
        especifi = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                   scoring=make_scorer(recall_score, average='macro'))
        val_scores_accuracy = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='accuracy')
        val_scores_mcc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,
                                         scoring=make_scorer(matthews_corrcoef))
        val_scores_roc = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5, scoring='roc_auc')
        # t = cross_val_score(modelo, self.x_trein, self.y_trein, cv=5,scoring=make_scorer(recall_score, labels=0))
        # print(np.mean(t))

        print(
            "Validação Interna: sensitivity {:.2f}%, specificity {:.2f}%, acurácia: {:.2f}%, MCC {:.2f}%, AUROC {:.2f}%.".format(
                np.mean(val_scores_sensitivity), np.mean(especifi), np.mean(val_scores_accuracy)*100,
                np.mean(val_scores_mcc), np.mean(val_scores_roc)))
        modelo.fit(self.x_trein, self.y_trein)
        predict = modelo.predict(self.x_val)
        accuracy = accuracy_score(self.y_val, predict)
        sensitivity = recall_score(self.y_val, predict)
        mcc = matthews_corrcoef(self.y_val, predict)
        roc = roc_auc_score(self.y_val, predict)
        tn, fp, fn, tp = confusion_matrix(self.y_val, predict).ravel()
        specificity = tn / (tn + fp)  # make_scorer(self.y_val, predict) #tn / (tn + fp)
        print(
            "Validação Externa: sensitivity {:.2f}%, specificity {:.2f}%,  acurácia {:.2f}%, MCC {:.2f}%, AUROC  {:.2f}%".format(
                sensitivity, specificity,
                accuracy*100,
                mcc, roc))


if __name__ == '__main__':
    m = Modelos_classificacao()
    #'hidden_layer_sizes': (2500, 100), 'learning_rate_init': 0.01, 'solver': 'sgd', 'learning_rate': 'adaptive'
    m.modelo_svm()
    m.modelo_RF()
    m.modelo_ETree()
    m.modelo_mlp()
    m.modelo_knn()
    m.modelo_Ridge()