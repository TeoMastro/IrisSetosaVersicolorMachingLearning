# -*- coding: utf-8 -*-

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data=read_csv(r'C:\Users\vagmark\.spyder-py3\iris.data', header=None).values

#temporary variables gia thn emfanish grafhmatwn
tmpSubplot=421
tmpLSESubplot=421
tmpADALSubplot=421

##################################################################################################
#------------------------------------FUNCTIONS---------------------------------------------------#
##################################################################################################

#####################################################
#------------------PERCEPTRON FUNCTIONS-------------#
#####################################################
#Kanw ena prediction me ta weights apo th train weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
        #ypologismos toy u toy kathe protypoy
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
    #tyxaia weights gia kathe protypo toy pinaka train
    weights = [0.0 for i in range(len(train[0]))]
    
    #algorithmos ekpaideushs gia ton perceptron apo diafaneies
    for epoch in range(n_epoch):
        for row in train:
            #klhsh ths predict gia na moy epistrepsei 1 h 0 , dhladh to y
            prediction = predict(row, weights)
            
            #ypologizw to error gia kathe protypo
            error = row[-1] - prediction
            
            #ypologismos toy weight toy sygkekrimenoy protypoy
            weights[0] = weights[0] + l_rate * error
            
            #an h e3odos einai diaforetikh apo ton stoxo
            if(prediction!=row[2]):
                for i in range(len(row)-1):
                    #diorthwsh ton synapsewn
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    
    #Pinakas gia na apothikeusoume ta prediction ta opoia tha emfanisoume sto grafhma
    predict_t=np.zeros(((train.shape[0]),1))
    i=0
    
    for row in train :
        
        #ulopoiw to predict kai to apothikeuw gia na to emfanisw parakatw sto grafhma
        prediction = predict(row,weights)
        predict_t[i]=prediction
        i=i+1
    
    #temporary pinakas ,ton xrhsimopoioume mono gia to grafhma, paei apo to 0 mexri megethos tou train set
    predict_test_view=np.zeros((train.shape[0]))
    for y in range((predict_test_view.shape[0])):
        predict_test_view[y]=y

    #emfanish sto idio grafhma expected kai predicted 
    plt.subplot(tmpSubplot)
    plt.scatter(predict_test_view[:], train[:,2], label= "actual", color= "blue",marker= ".", s=50) 
    plt.subplot(tmpSubplot)
    plt.scatter(predict_test_view[:], predict_t[:], label= "predict", color= "red",marker= ".", s=5)
    if(tmpSubplot==421):
        plt.title('predict-expected')
        plt.show()
    if(tmpSubplot==422):
        plt.title('predict-expected')

    return weights


def perceptron(xtrain,ttrain, l_rate, n_epoch):
    
    #apo ton pinaka xtrain pairnw ta x,y data poy me endiaferoun gia xwro 2 diastasewn kai ta apothikeuw ston pinaka train
    train=xtrain[:,[0,2]]
    
    #prosthetw ton pinaka stoxwn ston pinaka me ta x,y toy train
    train=np.hstack((train,ttrain))
    
    #ypologizw ta weights, dhladh ekpaideuw to diktyo
    weights = train_weights(train, l_rate, n_epoch)
    
    print("---------------Weights-----------------")
    print(weights)
    
    predictions = list()
    for epoch in range(n_epoch):
        for row in train:
            
            #afou ekpaideusa to diktyo, me ta ypologismena weights kanw predict gia kathe protypo toy test
            #kai ta apotelesmata ta apothikeuw sth lista predictions thn opoia kai gyrname
            prediction=predict(row,weights)
            predictions.append(prediction)
    
    #epistrefw th lista me ta predictions gia ta test protypa 
    return(predictions)

#####################################################
#------------TELOS PERCEPTRON FUNCTIONS-------------#
#####################################################
    

#####################################################
#-----------------ADALINE FUNCTIONS-----------------#
#####################################################
    
# Kanw ena prediction me ta weights apo th train_weights_adaline
def predictAdaline(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
        #ypologismos toy u toy kathe protypoy
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else -1.0



def train_weights_adaline(train,minmse, l_rate, n_epoch):
    
    #tyxaia weights gia kathe protypo toy pinaka train
    weights = [0.0 for i in range(len(train[0]))]
    
    for epoch in range(n_epoch): 
        for row in train:
            
            #klhsh ths predictAdaline gia na moy epistrepsei 1 h -1 , dhladh to y
            prediction = predictAdaline(row, weights)
            
            #ypologismos toy mesoy tetragwnikoy sfalmatos
            Jmse = (row[2]-prediction)**2
            
            #elegxos an to meso tetragwniko sfalma einai megalytero toy oriou minmse
            if(Jmse > minmse):
                
                #ypologizw to error gia kathe protypo
                error = row[-1] - prediction
                
                #ypologismos toy weight toy sygkekrimenoy protypoy
                weights[0] = weights[0] + l_rate * error
                
                for i in range(len(row)-1):
                    #diorthwsh ton synapsewn
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i] 
       
    #Pinakas gia na apothikeusoume ta prediction ta opoia tha emfanisoume sto grafhma             
    predict_t=np.zeros(((train.shape[0]),1))  
    i=0      
    for row in train :
        
        #ulopoiw to predict kai to apothikeuw gia na to emfanisw parakatw sto grafhma
        prediction = predictAdaline(row,weights)
        predict_t[i]=prediction
        i=i+1
        
    #temporary pinakas ,ton xrhsimopoioume mono gia to grafhma, paei apo to 0 mexri megethos tou train set   
    predict_test_view=np.zeros((train.shape[0]))
    for y in range((predict_test_view.shape[0])):
        predict_test_view[y]=y

    #emfanish sto idio grafhma expected kai predicted 
    plt.subplot(tmpADALSubplot)
    plt.scatter(predict_test_view[:], train[:,2], label= "actual", color= "blue",marker= ".", s=50) 
    plt.subplot(tmpADALSubplot)
    plt.scatter(predict_test_view[:], predict_t[:], label= "predict", color= "red",marker= ".", s=5) 
    if(tmpADALSubplot==421):
        plt.title('predict-expected')
        plt.show()
    if(tmpADALSubplot==422):
        plt.title('predict-expected') 

    return weights


def adaline(xtrain,ttrain,minmse, l_rate, n_epoch):
    
    #apo ton pinaka xtrain pairnw ta x,y data poy me endiaferoun gia xwro 2 diastasewn kai ta apothikeuw ston pinaka train
    train=xtrain[:,[0,2]]
    
    #prosthetw ton pinaka stoxwn ston pinaka me ta x,y toy train
    train=np.hstack((train,ttrain))
    
    #ypologizw ta weights, dhladh ekpaideuw to diktyo
    weights = train_weights_adaline(train,minmse, l_rate, n_epoch)
    
    print("---------------Weights-----------------")
    print(weights)
    
    predictions = list()
    for epoch in range(n_epoch):
        for row in train:
            
            #afou ekpaideusa to diktyo, me ta ypologismena weights kanw predict gia kathe protypo toy test kai ta apotelesmata ta apothikeuw sth lista predictions
            prediction=predictAdaline(row,weights)
            predictions.append(prediction)
            
    #epistrefw th lista me ta predictions gia ta test protypa 
    return(predictions)

#####################################################
#---------------TELOS ADALINE FUNCTIONS-------------#
#####################################################


#####################################################
#-------LUSI ELAXISTWN TETRAGWNWN FUNCTION----------#
#####################################################

#th xreiazomaste gia ta folds
def lushElaxistwnTetragwnwn():
    
    #allazw to ttrain kai ttest se -1 kai 1 apo 0 kai 1
    for i,x in enumerate(ttrain):
        if x==0: ttrain[i]=-1
    for i,x in enumerate(ttest):
        if x==0: ttest[i]=-1
        
    #apo ton xtest pairnw mono ta x kai y    
    xxtest=xtest[:,[0,2]].astype(float)
    
    #enwsh twn xxtest kai ttest dhladh enwsh twn x kai y me tous stoxous tous
    xxtest=np.hstack((xxtest,ttest))
    
    #apo ton xtrain pairnw mono ta x kai y 
    train=xtrain[:,[0,2]].astype(float)
    
    #enwsh twn train kai ttrain dhladh enwsh twn x kai y me tous stoxous tous
    train=np.hstack((train,ttrain))
    
    #ypologismos twn weights twn protypwn me th xrhsh tou eswterikoy ginomenou ths etoimhs synarthshs np.linalg.pinv() me ton ttrain
    weights = np.linalg.pinv(train).dot(ttrain)
    
    #ypologismos ths e3odou y me th xrhsh eswterikou ginomenoy meta3u twn pinakwn xxtest kai weights
    y = xxtest.dot(weights)

    predict_ = y
    
    #dhmiourgw mia empty lista me predictions 
    predictions = list()
    
    #kanw tous katallilous elegxous gia na vgalw to prediction 
    for i in range(len(y)):
        if y[i] < 0:
            predict_[i] = 0
            predictions.append(0)
        else:
            predict_[i] = 1
            predictions.append(1)

    #temporary pinakas ,ton xrhsimopoioume mono gia to grafhma, paei apo to 0 mexri megethos tou xxtest set   
    predict_test_view=np.zeros((xxtest.shape[0]))
    for y in range((predict_test_view.shape[0])):
        predict_test_view[y]=y
        
    #epanafora sta 0-1 mono kai mono gia na vgei swsto to grafhma
    for i,x in enumerate(ttest):
        if x==-1: ttest[i]=0
    
    #kai to apothikeuoume sto xxtest to opoio kai tha emfanisoume
    xxtest=xtest[:,[0,2]].astype(float)
    xxtest=np.hstack((xxtest,ttest))
    
    #emfanish sto idio grafhma expected kai predicted 
    plt.subplot(tmpLSESubplot)
    plt.scatter(predict_test_view[:], xxtest[:,2], label= "actual", color= "blue",marker= ".", s=50)
    plt.subplot(tmpLSESubplot)
    plt.scatter(predict_test_view[:], predict_[:], label= "predict", color= "red",marker= ".", s=5)
    if(tmpLSESubplot==421):
        plt.title('predict-expected')
        plt.show()
    if(tmpLSESubplot==422):
        plt.title('predict-expected') 

#####################################################
#----TELOS LUSI ELAXISTWN TETRAGWNWN FUNCTION-------#
#####################################################
        
##############################################################################
#----------------------------TELOS FUNCTIONS---------------------------------#
##############################################################################



##############################################################################
#--------------------XEKINAEI H ROH TOU PROGRAMMATOS-------------------------#
##############################################################################

#kanoume data manipulation pou tha mas voithhsei gia ta grafhmata kai parakatw
xdata=data[:,[0,1,2,3]]
xclass=data[:,[4]]

Setosa=np.hstack((xdata[0:50,[0]],xdata[0:50,[2]]))
Versicolor=np.hstack((xdata[50:100,[0]],xdata[50:100,[2]]))
Virginica=np.hstack((xdata[100:150,[0]],xdata[100:150,[2]]))


#Emfanish grafhmatos me ola ta protypa mesa
plt.scatter(Setosa[:,0], Setosa[:,1], label= "setosa", color= "blue",marker= ".", s=30)
plt.scatter(Versicolor[:,0], Versicolor[:,1], label= "versicolor", color= "red",marker= ".", s=30) 
plt.scatter(Virginica[:,0], Virginica[:,1], label= "virginica", color= "green",marker= ".", s=30)

plt.title('Grafhma olwn twn Protypwn') 
plt.legend() 
plt.show()

#pinakas t[patterns] <- stoxous
t=np.zeros((data.shape[0],1))

#menu gia epilogh ston diaxwrismo toy dataset
ans='y'
while (ans == 'y'):
    print("Epilexte diaxwrismo protypwn")
    print("----------------------------")
    print("1.Diaxwrismos Iris-setosa")
    print("2.Diaxwrismos Iris-virginica")
    print("3.Diaxwrismos Iris-versicolor")
    
    #kaname ta dict alla den ta xrhsimopoihsame
    input_1 = input("Dwse arithmo : ")
    if(input_1=='0'):
        ans='n'
    if(input_1=='1'):
        map_dict1={"Iris-setosa":1,"Iris-versicolor":0,"Iris-virginica":0}
        for i in range(150):
            if(data[i][4]=="Iris-setosa"):
                t[i]=1
            else:
                t[i]=0
    if(input_1=='2'):
        map_dict2={"Iris-setosa":0,"Iris-versicolor":0,"Iris-virginica":1}
        for i in range(150):
            if(data[i][4]=="Iris-virginica"):
                t[i]=1
            else:
                t[i]=0
    if(input_1=='3'):
        map_dict3={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":0}
        for i in range(150):
            if(data[i][4]=="Iris-versicolor"):
                t[i]=1
            else:
                t[i]=0
    print("Oloklhrwthike o diaxwrismos protypwn")
    print("------------------------------------")
    
    #epau3hsh toy pinaka twn data me 1 sto telos toy   
    temp=np.ones((150,1))
    xdata=np.hstack((xdata,temp))
    
##############################################################################
#--------XWRISMOS PROTUPWN SE EKPEUDEUSHS KAI ANAKLISIS----------------------#
##############################################################################
    
    #dhmiourgia toy pinaka xtrain
    xtrain=[]
    xtrainSetosa = np.vstack(xdata[0:40])
    xtrainVirginica = np.vstack(xdata[50:90])
    xtrainVersicolor = np.vstack(xdata[100:140])
    
    xtrainSetosaVerginica = np.vstack((xtrainSetosa,xtrainVirginica))
    xtrain = np.vstack((xtrainSetosaVerginica,xtrainVersicolor))
    
    #dhmiourgia toy pinaka xtest
    xtest=[]
    xtestSetosa = np.vstack(xdata[40:50])
    xtestVirginica = np.vstack(xdata[90:100])
    xtestVersicolor = np.vstack(xdata[140:150])
    
    xtestSetosaVerginica = np.vstack((xtestSetosa,xtestVirginica))
    xtest = np.vstack((xtestSetosaVerginica,xtestVersicolor))   
    
    
    #dhmiourgia toy dianusmatos ttrain , stoxwn pou tha xrhsimopoihthoyn sthn ekpaideush
    ttrain=[]
    xySetosa = t[0:40]
    xyVersicolor = t[50:90]
    xyVirginica = t[100:140]
    
    ttrainSetosaVersicolor=np.vstack((xySetosa,xyVersicolor))
    ttrain = np.vstack((ttrainSetosaVersicolor,xyVirginica))
    
    #dhmiourgia toy dianusmatos ttest , stoxoi pou tha xrhsimopoihthoun gia elegxo 
    ttest=[]
    xyLast10Setosa = t[40:50]
    xyLast10Versicolor = t[90:100]
    xyLast10Virginica = t[140:150]
    
    ttestSetosaVersicolor=np.vstack((xyLast10Setosa,xyLast10Versicolor))
    ttest = np.vstack((ttestSetosaVersicolor,xyLast10Virginica))
    
##############################################################################
#----------------------------TELOS XWRISMOU----------------------------------#
##############################################################################
    
    #emfanish tou grafhmatos xtrain kai xtest
    plt.scatter(xtrain[:,0], xtrain[:,2], label= "ttrain setosa,versicolor,virginica", color= "blue",marker= ".", s=30)
    plt.scatter(xtest[:,0], xtest[:,2], label= "ttest setosa,versicolor,virginica", color= "red", marker= ".", s=30)

    plt.title('Grafhma twn xtrain kai xtest') 
    plt.legend() 
    plt.show()
    
    
    #menu epiloghs methodou
    choice=0
    while (choice!=4):
        print("Epilexte algorithmo")
        print("-------------------")
        print("1.Ylopoihsh me Perceptron")
        print("2.Ylopoihsh me Adaline")
        print("3.Ylopoihsh me Lush Elaxistwn Tetragwnwn")
        print("4.Exit to starting menu")
        
        input_1 = input("Dwse arithmo : ")
        
        if(input_1=='4'):
            
            choice=4
            break
        
        if(input_1=='1'):
            #------------------------------------------------------------------------------------------------------------Perceptron
            choice=1

            input_Epochs=int(input("Dwse arithmo epanalhpsewn(maxEpochs): "))
            input_SyntelsthsEkpaideushs=float(input("Dwse arithmo Syntelsth Ekpaideushs(beta): "))
            
            #klhsh ths methodou perceptron
            perceptron(xtrain,ttrain,input_SyntelsthsEkpaideushs, input_Epochs)
            
            print("\n Xekinane ta folds \n")
            for f in range(4):
                
                #kalw thn TRAIN_TEST_SPLIT
                X_train,X_test,y_train,y_test=train_test_split(xtrain, ttrain, test_size=0.1)
                
                # h metavlhth tmpSubplot einai voithitikh gia ta grafhmata
                
                #emfanizw ta randomized dedomena san xtrain kai xtest
                plt.subplot(tmpSubplot)
                plt.scatter(X_train[:,0], X_train[:,2], label= "X_train", color= "blue",marker= ".", s=30) 
                plt.subplot(tmpSubplot)
                plt.scatter(X_test[:,0], X_test[:,2], label= "X_test", color= "red",marker= ".", s=30)
                
                if(tmpSubplot==421):#titlos mono sto prwto grafhma
                    plt.title('xtrain - xtest') 
                tmpSubplot=tmpSubplot+1
                
                #kalw thn perceptron me tous kainourgious pinakes pou mas edwse h train_test_split
                perceptron(X_train,y_train,input_SyntelsthsEkpaideushs, input_Epochs)
                
                tmpSubplot=tmpSubplot+1
            
            #meta th for emfanise ta grafhmata   
            plt.show()
            
            #epanaferw thn timh gia na vgenoun swsta ta grafhmata an xanakalesw thn idia func
            tmpSubplot=421
            
        if(input_1=='2'):
            choice=2
            #------------------------------------------------------------------------------------------------------------Adaline

            #allazw to ttrain kai ttest se -1 kai 1 apo 0 kai 1
            for i,x in enumerate(ttrain):
                if x==0: ttrain[i]=-1
            for i,x in enumerate(ttest):
                if x==0: ttest[i]=-1
            
            #inputs
            input_Epochs=int(input("Dwse arithmo epanalhpsewn(maxEpochs): "))
            input_SyntelsthsEkpaideushs=float(input("Dwse arithmo Syntelsth Ekpaideushs(beta): "))
            input_Minmse=float(input("Dwse minmse: "))
            
            #klhsh ths methodoy adaline
            adaline(xtrain,ttrain,input_Minmse,input_SyntelsthsEkpaideushs, input_Epochs)
            
            print("\n Xekinane ta folds \n")
            
            for f in range(4):
                
                #kalw thn TRAIN_TEST_SPLIT
                X_train,X_test,y_train,y_test=train_test_split(xtrain, ttrain, test_size=0.1)
                
                # h metavlhth tmpADALSubplot einai voithitikh gia ta grafhmata
                
                #emfanizw ta randomized dedomena san xtrain kai xtest
                plt.subplot(tmpADALSubplot)
                plt.scatter(X_train[:,0], X_train[:,2], label= "X_train", color= "blue",marker= ".", s=30) 
                plt.subplot(tmpADALSubplot)
                plt.scatter(X_test[:,0], X_test[:,2], label= "X_test", color= "red",marker= ".", s=30)
                
                if(tmpADALSubplot==421):#titlos mono sto prwto grafhma
                    plt.title('xtrain - xtest') 
                tmpADALSubplot=tmpADALSubplot+1
                
                #kalw thn adaline me tous kainourgious pinakes pou mas edwse h train_test_split
                adaline(X_train,y_train,input_Minmse,input_SyntelsthsEkpaideushs, input_Epochs)
                
                tmpADALSubplot=tmpADALSubplot+1
            
            #meta th for emfanise ta grafhmata 
            plt.show()
            
            #epanaferw thn timh gia na vgenoun swsta ta grafhmata an xanakalesw thn idia func
            tmpADALSubplot=421

        if(input_1=='3'):
            choice=3
            #------------------------------------------------------------------------------------------------------------Lush Elaxistwn Tetragwnwn
            
            #klhsh ths methodoy lushElaxistwnTetragwnwn
            lushElaxistwnTetragwnwn()
            
            for f in range(4):
                
                #kalw thn TRAIN_TEST_SPLIT
                X_train,X_test,y_train,y_test=train_test_split(xtrain, ttrain, test_size=0.1)
                
                # h metavlhth tmpLSESubplot einai voithitikh gia ta grafhmata
                
                #emfanizw ta randomized dedomena san xtrain kai xtest
                plt.subplot(tmpLSESubplot)
                plt.scatter(X_train[:,0], X_train[:,2], label= "X_train", color= "blue",marker= ".", s=30) 
                plt.subplot(tmpLSESubplot)
                plt.scatter(X_test[:,0], X_test[:,2], label= "X_test", color= "red",marker= ".", s=30)
                
                if(tmpLSESubplot==421):#titlos mono sto prwto grafhma
                    plt.title('xtrain - xtest') 
                tmpLSESubplot=tmpLSESubplot+1
                
                #kalw thn lushElaxistwnTetragwnwn
                lushElaxistwnTetragwnwn()
                
                tmpLSESubplot=tmpLSESubplot+1
            
            #meta th for emfanise ta grafhmata 
            plt.show()
            #epanaferw thn timh gia na vgenoun swsta ta grafhmata an xanakalesw thn idia func
            tmpLSESubplot=421
    
    ans = input("Synexizoume y/n : ")
##############################################################################
#----------------------------TELOS PROGRAMMATOS------------------------------#
##############################################################################