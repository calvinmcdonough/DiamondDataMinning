from catboost import Pool, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# working
trainSet = pd.read_csv("train.csv")
testSet = pd.read_csv("test.csv")
#sm = pd.read_csv("sm.csv")
sample = pd.read_csv("sample_submission.csv")

arrayTrainSet = np.array(trainSet)
numColumns = len(trainSet.columns)
#smArray = np.array(sm)

attributes = np.array(arrayTrainSet[:,0:numColumns-1])
values = np.array(arrayTrainSet[:,numColumns-1])
#print(smArray)

x_train, x_test, y_train, y_test = train_test_split(attributes, values, test_size=0.2, random_state=0)

list_of_att =[]

arrayTest = np.array(testSet)
arraySample = np.array(sample)

print(len(arrayTest))
print(len(arraySample))


for atribute in x_train:
    #print(atribute[4])
    if (atribute[2] in list_of_att) == False:
        list_of_att.append(atribute[2])
    if atribute[2] == "Ideal":
        atribute[2]=0
    if atribute[2] == "Very Good":
        atribute[2]=1
    if atribute[2] == "Premium":
        atribute[2]=2
    if atribute[2] == "Good":
        atribute[2]=0
    if atribute[2] == "Fair":
        atribute[2]=0
        
    if atribute[3] == "E":
        atribute[3]=0
    if atribute[3] == "J":
        atribute[3]=1
    if atribute[3] == "G":
        atribute[3]=2
    if atribute[3] == "F":
        atribute[3]=3
    if atribute[3] == "I":
        atribute[3]=4
    if atribute[3] == "D":
        atribute[3] = 5
    if atribute[3] == "H":
        atribute[3] = 6
        
    if atribute[4] == "VS2":
        atribute[4] = 0
    if atribute[4] == "SI2":
        atribute[4] = 1
    if atribute[4] == "VS1":
        atribute[4] = 2
    if atribute[4] == "SI1":
        atribute[4] = 3
    if atribute[4] == "VVS2":
        atribute[4] = 4
    if atribute[4] == "VVS1":
        atribute[4] = 5
    if atribute[4] == "IF":
        atribute[4] = 6
    if atribute[4] == "I1":
        atribute[4] = 7
   
for atribute in arrayTest:
    
    
    if atribute[2] == "Ideal":
        atribute[2]=0
    if atribute[2] == "Very Good":
        atribute[2]=1
    if atribute[2] == "Premium":
        atribute[2]=2
    if atribute[2] == "Good":
        atribute[2]=0
    if atribute[2] == "Fair":
        atribute[2]=0
        
    if atribute[3] == "E":
        atribute[3]=0
    if atribute[3] == "J":
        atribute[3]=1
    if atribute[3] == "G":
        atribute[3]=2
    if atribute[3] == "F":
        atribute[3]=3
    if atribute[3] == "I":
        atribute[3]=4
    if atribute[3] == "D":
        atribute[3] = 5
    if atribute[3] == "H":
        atribute[3] = 6
        
    if atribute[4] == "VS2":
        atribute[4] = 0
    if atribute[4] == "SI2":
        atribute[4] = 1
    if atribute[4] == "VS1":
        atribute[4] = 2
    if atribute[4] == "SI1":
        atribute[4] = 3
    if atribute[4] == "VVS2":
        atribute[4] = 4
    if atribute[4] == "VVS1":
        atribute[4] = 5
    if atribute[4] == "IF":
        atribute[4] = 6
    if atribute[4] == "I1":
        atribute[4] = 7



m = CatBoostRegressor()
m.fit(x_train, y_train)

preds_class = m.predict(arrayTest)

print("class = ", preds_class)
pd.DataFrame(preds_class).to_csv("submit2.csv")

print(len(preds_class))