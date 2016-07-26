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

#Set the seed
random.seed(42)

filepath_train="./data/ech_apprentissage.csv"
filepath_test="./data/ech_test.csv"

vartype = [
        None, # id
        "NUM", # annee_naissance
        "NUM", # annee_permis
        "ONEHOT", # marque
        "NUM", # puis_fiscale
        "NUM", # anc_veh
        None, # codepostal
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
        None, # var14
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

train_dataset = open_and_refine_data(filepath_train)
test_dataset = open_and_refine_data(filepath_test)
my_ids = extract_id(test_dataset)

train0 = extract_predictor(train_dataset, True)
train1 = extract_target(train_dataset)
test0 = extract_predictor(test_dataset, False)

results = []
for cnt in range(1000):
    projected0 = []
    projected1 = []
    for i in xrange(len(train0)):
        if random.random() < 0.4:
            continue
        projected0.append(train0[i])
        projected1.append(train1[i])
    print "now fitting the model", cnt, "with len", len(projected0)
    model = Ridge()
    model.fit(projected0, projected1)
    predictions=model.predict(test0)
    results.append(list(predictions))

final_result = []
for ind in xrange(len(results[0])):
    cand = []
    for i in xrange(len(results)):
        cand.append(results[i][ind])
    final_result.append(sum(sorted(cand)[100:-100])*1.0/(len(cand)-200))

#predictions=model.predict(valid_dataset)

#Evaluate the quality of the prediction
#print sklearn.metrics.mean_absolute_error(predictions,valid_target)

print "saving the predicted result into the file"
f = open('result.csv', 'w')
f.write("ID;COTIS\n");
for ind, prd in enumerate(final_result):
    f.write(my_ids[ind] + ';' + str(prd) + '\n')
f.close()
print "all tasks completed"
