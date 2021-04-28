import sys
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
import matplotlib.pyplot
import numpy as np
from sklearn.linear_model import Perceptron
import progressbar




def main(dataTrain, dataTest):
    # loads data
    print ("Loading data...")
    X_train, y_train = load_svmlight_file(dataTrain)
    X_test, y_test = load_svmlight_file(dataTest)
    
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    blocos = [100,200,400,600,800,1000]
    k=1000
    while(k<20000):
        k+=1000
        blocos.append(k)
    print(blocos)
    
    resultados = []

    for bloco in progressbar(blocos):

        X_train_new = X_train[:bloco]
        Y_train_new = y_train[:bloco]
    
        clf = perceptron(n_jobs=-1).fit(X_train_new,Y_train_new)
    
        # mostra o resultado do classificador na base de teste
        accuracy.append(clf.score(X_test, y_test))            
    
        #predict para Matriz de confusão
        pred = clf.predict(X_test)
            
    
        # cria a matriz de confusao
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, labels=[0,1,2,3,4,5,6,7,8,9]))
        print ('Accuracy: ',  clf.score(X_test, y_test),'\n')

    #plotar gráficos
    print("ACURACIA",accuracy)
    matplotlib.pyplot.title('KNN', fontsize=15 )
    matplotlib.pyplot.xlabel('Base',fontsize=15)
    matplotlib.pyplot.ylabel('Acurácia',fontsize=15)
    matplotlib.pyplot.plot(blocos, accuracy, color = 'r', linewidth = '2')
    matplotlib.pyplot.show()
    
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
     




