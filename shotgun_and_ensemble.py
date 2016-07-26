import csv
import random
from sklearn import naive_bayes
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.ridge import Ridge
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import sys
import json

#Set the seed
random.seed(42)

if len(sys.argv) == 1:
    print "you should specify argv"
    sys.exit()

filepath_train="./data/ech_apprentissage.csv"
filepath_test="./data/ech_test.csv"

vartype = [
        None, # id
        "NUM", # annee_naissance
        "NUM", # annee_permis
        "ONEHOT", # marque
        "NUM", # puis_fiscale
        "NUM", # anc_veh
        "NUM", # codepostal
        "ONEHOT", # energie_veh
        "NUM", # kmage_annuel
        "NUM", # crm
        "ONEHOT", # profession
        "NUM", # var1
        "ONEHOT", # var2
        "ONEHOT", # var3
        "ONEHOT", # var4
        "ONEHOT", # var5
        "ONEHOT", # var6
        "NUM", # var7
        "ONEHOT", # var8
        "NUM", # var9
        "NUM", # var10
        "NUM", # var11
        "NUM", # var12
        "ONEHOT", # var13
        "ONEHOT", # var14
        "ONEHOT", # var15
        "ONEHOT", # var16
        "ONEHOT", # var17
        "NUM", # var18
        "NUM", # var19
        "ONEHOT", # var20
        "ONEHOT", # var21
        "NUM", # var22
        ]

varinfo = []
for i in range(len(vartype)):
    varinfo.append(None)

num_var = len(vartype)

def open_and_refine_data(filepath):
    file_open=open(filepath,"r")
    csv_reader=csv.reader(file_open,delimiter=";")
    header=csv_reader.next()
    dataset=[]
    for row in csv_reader:
        dataset.append(row)
    for index,row in enumerate(dataset):
        dataset[index]=[value if value not in ["NR",""] else -1 for value in row]
    return dataset

def extract_predictor(dataset, flag):
    dataset_extracted = []
    for i in xrange(len(dataset)):
        dataset_extracted.append([])
    for ind in range(len(vartype)):
        if vartype[ind] == "NUM":
            print "working with variable", ind, "which is NUM"
            for i in xrange(len(dataset)):
                try:
                    dataset_extracted[i].append(float(dataset[i][ind]))
                except Exception:
                    dataset_extracted[i].append(-1)
        print dataset_extracted[0]
    for ind in range(len(vartype)):
        if vartype[ind] == "ONEHOT":
            print "working with variable", ind, "which is ONEHOT"
            if flag:
                print "count distinct elements"
                varinfo[ind] = set()
                for i in xrange(len(dataset)):
                    varinfo[ind].add(dataset[i][ind])
                varinfo[ind] = list(varinfo[ind])
                print "completed"
            for i in xrange(len(dataset)):
                temp = [0] * (len(varinfo[ind]) + 1)
                d = dataset[i][ind]
                if d in varinfo[ind]:
                    temp[varinfo[ind].index(d)] = 1
                else:
                    temp[-1] = 1
                dataset_extracted[i] += temp
        print dataset_extracted[0]
    return dataset_extracted

def extract_target(dataset):
    targets=[]
    for row in dataset:
        targets.append(float(row[-1]))
    return targets

def extract_id(dataset):
    ids = []
    for row in dataset:
        ids.append(row[0])
    return ids

def train_and_predict(train0, train1, test0):
    models = [
            ExtraTreesRegressor,
            RandomForestRegressor,
            GradientBoostingRegressor,
            #GaussianNB,
            Ridge,
            KNeighborsRegressor,
            DecisionTreeRegressor
            ]

    model_names = [
            "ExtraTreesRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            #"GaussianNB",
            "Ridge",
            "KNeighborsRegressor",
            "DecisionTreeRegressor"
            ]

    results = []
    for model, name in zip(models, model_names):
        my_model = model()
        print "now fitting the model", name
        my_model.fit(train0, train1)
        print "start prediction"
        predictions=my_model.predict(test0)
        results.append(list(predictions))

    transposed = []
    for ind in xrange(len(results[0])):
        predicted = []
        for i in xrange(len(results)):
            predicted.append(results[i][ind])
        transposed.append(predicted)

    return transposed

train_dataset = open_and_refine_data(filepath_train)
test_dataset = open_and_refine_data(filepath_test)
my_ids = extract_id(test_dataset)

if sys.argv[1] == "first":
    print "make input data"
    train0 = extract_predictor(train_dataset, True)
    train1 = extract_target(train_dataset)
    test0 = extract_predictor(test_dataset, False)
    print "save the input data"
    ff = open('input_refined.json', 'w')
    ff.write(json.dumps([train0, train1, test0]))
    ff.close()
    print "completed"

    print "Training and predicting with whole data"
    intermediate_results = train_and_predict(train0, train1, test0)
    print "saving the intermediate result"
    ff = open("intermediate_result.json", "w")
    ff.write(json.dumps([intermediate_results, train1]))
    ff.close()
    print "completed"

    print "start making virtual prediction data"

    model2_train0 = []
    model2_train1 = []

    for cnt in range(3):
        virtual_train0 = []
        virtual_train1 = []
        virtual_test0 = []
        virtual_test1 = []

        for i in xrange(len(train0)):
            if random.random() < 0.7:
                virtual_train0.append(train0[i])
                virtual_train1.append(train1[i])
            else:
                virtual_test0.append(train0[i])
                virtual_test1.append(train1[i])
        print "start virtual training no.", cnt, "with", len(virtual_train0), "train data and", len(virtual_test0), "test data"
        model2_train0 += train_and_predict(virtual_train0, virtual_train1, virtual_test0)
        model2_train1 += virtual_test1

    print "now saving the result"

    ff = open('virtual_train_data.json', 'w')
    ff.write(json.dumps([model2_train0, model2_train1]))
    ff.close()

if sys.argv[1] == "second":
    ff = open('virtual_train_data.json', 'r')
    model2_train0, model2_train1 = json.loads(ff.read())
    ff.close()
    print "opened train0 and train1 with each length", len(model2_train0), len(model2_train1)
    print model2_train0[0]
    print model2_train1[0]
    ff = open('intermediate_result.json', 'r')
    model2_test0, _ = json.loads(ff.read())
    print model2_test0[0]
    model2 = Ridge()
    print "start fitting 2nd model"
    model2.fit(model2_train0, model2_train1)
    print "start predicting"
    predictions=model2.predict(model2_test0)
    print "saving the predicted result into the file"
    f = open('result.csv', 'w')
    f.write("ID;COTIS\n");
    for ind, prd in enumerate(predictions):
        f.write(my_ids[ind] + ';' + str(prd) + '\n')
    f.close()
    print "all tasks completed"
