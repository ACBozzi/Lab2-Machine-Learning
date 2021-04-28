import numpy
from sklearn import preprocessing
import pylab as pl
import matplotlib.pyplot
import numpy as np


        
blocos = [100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

perceptron = [0.5412815878320772, 0.7170651024792825, 0.755601405040412, 0.753708692834976, 0.708010776523548, 0.7560788459570985, 0.8801623299116734, 0.8987654742011391, 0.8517545953688231, 0.8978787982130069, 0.8805204105991884, 0.9073594107014972, 0.890171537700781, 0.9184940149370802, 0.8603485318691811, 0.9384442246700542, 0.9051768236537872, 0.9282474508065341, 0.8981857245165911, 0.9120656140231218, 0.906336323022883, 0.9353238072502813, 0.9339426388841524, 0.9223305937318829, 0.9269856426695768]


knn = [0.5798860962384477, 0.6762439041025816, 0.7211915561163592, 0.7510316134092692, 0.7637690550080142, 0.7819970671486547, 0.8587456945060191, 0.8856870033761893, 0.8897111482454046, 0.8939228591890325, 0.9168741261126079, 0.9247860041605566, 0.9292875899464584, 0.9326296763632643, 0.9348463663335947, 0.9344371312621491, 0.9355113733246939, 0.9357841967056577, 0.9356989393991065, 0.9353408587115916, 0.9370630563039253, 0.9368925416908229, 0.9373017767622686, 0.9382225556730212, 0.9385465334379156]

gaus = [0.5399515738498789, 0.4723425297548, 0.5237356341438462, 0.5437881526446816, 0.5860416737714422, 0.6743511918971455, 0.7820141186099648, 0.8384203526242199, 0.8448999079221089, 0.855096681785629, 0.8693517034409849, 0.8742454728370221, 0.8824813286498653, 0.8873580465845923, 0.889080244176926, 0.887409200968523, 0.8863861132899089, 0.8887051120281008, 0.8854653343791563, 0.884425195239232, 0.8858916209119122, 0.8864372676738397, 0.8880571564983119, 0.8885175459536883, 0.889080244176926]

logistic = [0.550728097397947, 0.6065375302663438, 0.6427548340892815, 0.6969955325171368, 0.7155645738839819, 0.7291204856256182, 0.782457456604031, 0.8068922006615967, 0.8138662483374826, 0.8190157896531732, 0.8422057770350919, 0.8632643317532313, 0.8809466971319442, 0.8886369061828598, 0.8960201889301913, 0.895781468471848, 0.8964464754629472, 0.8988507315076902, 0.898253930361832, 0.8982027759779013, 0.9021246120792552, 0.9047675885823415, 0.9080585206152167, 0.9100876445111346, 0.9118268935647785]

lda = [0.4994543532380725, 0.6373836237765577, 0.656549466289261, 0.656259591446987, 0.6987006786481601, 0.7873171230774477, 0.8732735395423388, 0.8935477270402074, 0.8853630256112949, 0.8905296183882959, 0.9025167956893906, 0.9124407461719469, 0.9207618592913412, 0.9237117620980118, 0.9278552671963988, 0.9260648637588241, 0.9245472837022133, 0.9242233059373188, 0.9220407188896088, 0.9208300651365822, 0.9210517341336153, 0.923149063874774, 0.9250588275415204, 0.9264229444463391, 0.9278552671963988]

#plotar gráficos
matplotlib.pyplot.title('Comparação', fontsize=15 )
matplotlib.pyplot.xlabel('Base',fontsize=15)
matplotlib.pyplot.ylabel('Acuracia',fontsize=15)


#plotar tamanho <1000 pela acurácia
matplotlib.pyplot.title('Classificação em função da base de treinamento', fontsize=15 )
matplotlib.pyplot.xlabel('Base',fontsize=15)
matplotlib.pyplot.ylabel('Acurácia',fontsize=15)
matplotlib.pyplot.plot(blocos, perceptron, color = 'red', label = 'Perceptron')
matplotlib.pyplot.plot(blocos, knn, color = 'green', label = 'Knn')
matplotlib.pyplot.plot(blocos, gaus, color = 'blue', label = 'GaussianNB')
matplotlib.pyplot.plot(blocos, logistic, color = 'pink', label = 'LogisticRegression')
matplotlib.pyplot.plot(blocos, lda, color = 'orange', label = 'LinearDiscrimantAnalysis')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

#classificadores=['Perceptron','LDiscAnal','LogRegres','GaussianNB','Knn']
#tempo = [11,12,15,17,193]
#
#acuracia = [94,93,91,89,93]
#
#matplotlib.pyplot.title('Acurácia para os 58 mil dados', fontsize=15 )
#matplotlib.pyplot.xlabel('Classificadores',fontsize=15)
#matplotlib.pyplot.ylabel('Acurácia',fontsize=15)
#matplotlib.pyplot.scatter(classificadores, acuracia, color = 'r', linewidth = '2')
#matplotlib.pyplot.show()
#


#plotar acuracia com 58000
#matplotlib.pyplot.title('Melhor tempo para 58000 dados', fontsize=15 )
#matplotlib.pyplot.xlabel('Base',fontsize=15)
#matplotlib.pyplot.ylabel('Tempo/s',fontsize=15)
#matplotlib.pyplot.scatter(58000, 11, color = 'red', label = 'Percept')#94
#matplotlib.pyplot.scatter(58000,193 , color = 'green', label = 'Knn')#93
#matplotlib.pyplot.scatter(58000, 17, color = 'blue', label = 'GaussNB') #89
#matplotlib.pyplot.scatter(58000, 15, color = 'pink', label = 'LogRegres')#91.
#matplotlib.pyplot.scatter(58000, 12, color = 'orange', label = 'LinearDiscrimantAnalysis')#93
#matplotlib.pyplot.legend()
#matplotlib.pyplot.show()
