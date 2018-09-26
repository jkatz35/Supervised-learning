import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.2
SEED = round(time.time() % 10)
#set the TEST_SET to 'leaf' for leaf data or 'bank' for bank data
TEST_SET = 'leaf'

TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5]

if (TEST_SET == 'leaf'):
    CROSS_VAL_SIZE = 5
else:
    CROSS_VAL_SIZE = 10

# Load dataset
if (TEST_SET == 'leaf'):
    leaf = pd.read_csv(r'leaf.csv')
    leaf_input = leaf.iloc[:, 2:]
    leaf_output = leaf.iloc[:, 0]
else:
    bank = pd.read_csv(r'data_banknote_authentication.csv')
    bank_input = bank.iloc[:, 0:-1]
    bank_output = bank.iloc[:, -1]

#split up dataset into training and test data
if (TEST_SET == 'leaf'):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(leaf_input, leaf_output, test_size=TEST_SIZE, random_state=SEED)
    X_train_1, X_test_1, Y_train_1, Y_test_1 = model_selection.train_test_split(leaf_input, leaf_output, test_size=.3, random_state=SEED)
    X_train_2, X_test_2, Y_train_2, Y_test_2 = model_selection.train_test_split(leaf_input, leaf_output, test_size=.4, random_state=SEED)
    X_train_3, X_test_3, Y_train_3, Y_test_3 = model_selection.train_test_split(leaf_input, leaf_output, test_size=.5, random_state=SEED)
    X_train_0, X_test_0, Y_train_0, Y_test_0 = model_selection.train_test_split(leaf_input, leaf_output, test_size=.1, random_state=SEED)
else:
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(bank_input, bank_output, test_size=TEST_SIZE, random_state=SEED)
    X_train_1, X_test_1, Y_train_1, Y_test_1 = model_selection.train_test_split(bank_input, bank_output, test_size=.3, random_state=SEED)
    X_train_2, X_test_2, Y_train_2, Y_test_2 = model_selection.train_test_split(bank_input, bank_output, test_size=.4, random_state=SEED)
    X_train_3, X_test_3, Y_train_3, Y_test_3 = model_selection.train_test_split(bank_input, bank_output, test_size=.5, random_state=SEED)
    X_train_0, X_test_0, Y_train_0, Y_test_0 = model_selection.train_test_split(bank_input, bank_output, test_size=.1, random_state=SEED)

X_trains = [X_train_0, X_train, X_train_1, X_train_2, X_train_3]
Y_trains = [Y_train_0, Y_train, Y_train_1, Y_train_2, Y_train_3]
X_tests = [X_test_0, X_test, X_test_1, X_test_2, X_test_3]
Y_tests = [Y_test_0, Y_test, Y_test_1, Y_test_2, Y_test_3]

#Decision Trees
prev_time = time.time()
optimal_depth = 0
max_score = 0
scores = []
train_accuracy = []
test_accuracy = []
depths = range(1, 50)
for depth in depths:
    #create Decision Tree Classifier and compute cross validation score
    clf = DecisionTreeClassifier(max_depth = depth)
    score = model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean()
    clf = clf.fit(X_train, Y_train)
    train_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    scores.append(score)
    if score > max_score:
        max_score = score
        optimal_depth = depth

plt.plot(depths, scores, label='Cross Validation Scores')
plt.plot(depths, train_accuracy, label='Training Accuracy Scores')
plt.plot(depths, test_accuracy, label='Testing Accuracy Scores')
plt.xlabel('Decision Tree Depth')
plt.ylabel('Accuracy')
plt.title("Decision Tree Performance on " + TEST_SET + " Data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_decisiontree.png')
plt.close()

#Create model with established optimal hyperparameters
clf = DecisionTreeClassifier(max_depth = optimal_depth)
clf = clf.fit(X_train, Y_train)
training_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)
predictions = clf.predict(X_test)

train_errors = []
test_errors = []
for i in range(len(TEST_SIZES)):
    clf = DecisionTreeClassifier(max_depth = optimal_depth)
    clf = clf.fit(X_trains[i], Y_trains[i])
    train_errors.append(clf.score(X_trains[i], Y_trains[i]))
    test_errors.append(clf.score(X_tests[i], Y_tests[i]))

plt.plot(TEST_SIZES, train_errors, label='Training Accuracy Scores')
plt.plot(TEST_SIZES, test_errors, label='Testing Accuracy Scores')
plt.xlabel('Test Data Size (%)')
plt.ylabel('Accuracy')
plt.title("Performance vs. Data Size on " + TEST_SET + " Data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_datasize.png')
plt.close()

file_name = TEST_SET + '//' + TEST_SET + '_decisiontree.txt'
with open(file_name, 'w') as f:
    print('Training accuracy decision tree: ', training_accuracy, file=f)
    print("test accuracy decision tree ", test_accuracy, file=f)
    print("optimal depth", optimal_depth, file=f)
    print("confusion matrix ", confusion_matrix(Y_test, predictions), file=f)
print(time.time() - prev_time)
prev_time = time.time()
#AdaBoost
optimal_estimators = 0
optimal_depth = 0
max_score = 0
estimator_list = range(5, 101, 5)
depths = range(1, 5)
for estimators in estimator_list:
    for depth in depths:
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = depth), n_estimators = estimators)
        score = model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean()
        if score > max_score:
            max_score = score
            optimal_estimators = estimators
            optimal_depth = depth

scores = []
train_accuracy = []
test_accuracy = []
for estimators in estimator_list:
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = optimal_depth), n_estimators = estimators)
    scores.append(model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean())
    clf = clf.fit(X_train, Y_train)
    train_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))

plt.plot(estimator_list, scores, label='Cross Validation Scores')
plt.plot(estimator_list, train_accuracy, label='Training Accuracy Scores')
plt.plot(estimator_list, test_accuracy, label='Testing Accuracy Scores')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title("AdaBoost Performance on " + TEST_SET + " data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_adaboost.png')
plt.close()

clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = optimal_depth), n_estimators = optimal_estimators)
clf = clf.fit(X_train, Y_train)
training_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)
predictions = clf.predict(X_test)

file_name = TEST_SET + '//' + TEST_SET + '_adaboost.txt'
with open(file_name, 'w') as f:
    print('Training accuracy adaboost: ', training_accuracy, file=f)
    print("test accuracy adaboost ", test_accuracy, file=f)
    print("optimal estimators ", optimal_estimators, file=f)
    print("optimal depth", optimal_depth, file=f)
    print("max score ", max_score, file=f)
    print("confusion matrix ", confusion_matrix(Y_test, predictions), file=f)

print(time.time() - prev_time)
prev_time = time.time()
#K-nearest neighbors

train_accuracy = []
test_accuracy = []
optimal_k = 0
max_score = 0
scores = []
k_list = range(1, 100)
for k in k_list:
    clf = KNeighborsClassifier(n_neighbors = k)
    score = model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean()
    scores.append(score)
    clf = clf.fit(X_train, Y_train)
    train_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    if score > max_score:
        max_score = score
        optimal_k = k

plt.plot(k_list, scores, label='Cross Validation Scores')
plt.plot(k_list, train_accuracy, label='Training Accuracy Scores')
plt.plot(k_list, test_accuracy, label='Testing Accuracy Scores')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title("KNN Performance " + TEST_SET + " data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_knn.png')
plt.close()

clf = KNeighborsClassifier(n_neighbors = optimal_k)
clf = clf.fit(X_train, Y_train)
training_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)
predictions = clf.predict(X_test)

file_name = TEST_SET + '//' + TEST_SET + '_knn.txt'
with open(file_name, 'w') as f:
    print('Training accuracy knn: ', training_accuracy, file=f)
    print("test accuracy knn ", test_accuracy, file=f)
    print("optimal neighbors ", optimal_k, file=f)
    print("max score ", max_score, file=f)
    print("confusion matrix ", confusion_matrix(Y_test, predictions), file=f)

print(time.time() - prev_time)
prev_time = time.time()
#Support Vector Machine
train_accuracy = []
test_accuracy = []
optimal_C = 0
max_score = 0
scores = []
C_list = [x / 5.0 for x in range(1, 50)]
for C in C_list:
    clf = SVC(kernel='linear', C=C)
    score = model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean()
    scores.append(score)
    clf = clf.fit(X_train, Y_train)
    train_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    if score > max_score:
        max_score = score
        optimal_C = C

plt.plot(C_list, scores, label='Cross Validation Scores')
plt.plot(C_list, train_accuracy, label='Training Accuracy Scores')
plt.plot(C_list, test_accuracy, label='Testing Accuracy Scores')
plt.xlabel('C Penalty Value')
plt.ylabel('Accuracy')
plt.title("SVM Performance " + TEST_SET + " data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_svm.png')
plt.close()

clf = SVC(kernel='linear', C=optimal_C)
clf = clf.fit(X_train, Y_train)
training_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)
predictions = clf.predict(X_test)

file_name = TEST_SET + '//' + TEST_SET + '_svm.txt'
with open(file_name, 'w') as f:
    print('Training accuracy svm: ', training_accuracy, file=f)
    print("test accuracy svm ", test_accuracy, file=f)
    print("optimal C ", optimal_C, file=f)
    print("max score ", max_score, file=f)
    print("confusion matrix ", confusion_matrix(Y_test, predictions), file=f)

print(time.time() - prev_time)
prev_time = time.time()
#Neural Network

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

train_accuracy = []
test_accuracy = []
scores = []
max_score = 0
optimal_layers = 1
optimal_layer_depth = 0
layer_list = range(1,5)
layer_depths = range(2,30)

for layer_depth in layer_depths:
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_depth), random_state=1)
    score = model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean()
    scores.append(score)
    clf = clf.fit(X_train, Y_train)
    train_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    if score > max_score:
        max_score = score
        optimal_layer_depth = layer_depth

plt.plot(layer_depths, scores, label='Cross Validation Scores')
plt.plot(layer_depths, train_accuracy, label='Training Accuracy Scores')
plt.plot(layer_depths, test_accuracy, label='Testing Accuracy Scores')
plt.xlabel('Layer Depth')
plt.ylabel('Accuracy')
plt.title("Neural Network Performance " + TEST_SET + " data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_nn1.png')
plt.close()

train_accuracy = []
test_accuracy = []
scores = []
max_score = 0
for layers in layer_list:
    layer_structure = [optimal_layer_depth for x in range(layers)]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_structure), random_state=1)
    score = model_selection.cross_val_score(clf, X_train, Y_train, cv = CROSS_VAL_SIZE).mean()
    scores.append(score)
    clf = clf.fit(X_train, Y_train)
    train_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    if score > max_score:
        max_score = score
        optimal_layers = layers

plt.plot(layer_list, scores, label='Cross Validation Scores')
plt.plot(layer_list, train_accuracy, label='Training Accuracy Scores')
plt.plot(layer_list, test_accuracy, label='Testing Accuracy Scores')
plt.xlabel('Layers')
plt.ylabel('Accuracy')
plt.title("Neural Network Performance " + TEST_SET + " data")
plt.legend()
plt.savefig(TEST_SET + '//' + TEST_SET + '_nn2.png')
plt.close()

layer_structure = [optimal_layer_depth for x in range(optimal_layers)]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_structure), random_state=1)
clf.fit(X_train,Y_train)
training_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)
predictions = clf.predict(X_test)

file_name = TEST_SET + '//' + TEST_SET + '_nn.txt'
with open(file_name, 'w') as f:
    print('Training accuracy NN: ', training_accuracy, file=f)
    print("test accuracy NN ", test_accuracy, file=f)
    print("optimal layer depth ", optimal_layer_depth, file=f)
    print("optimal layers", optimal_layers, file=f)
    print("max score ", max_score, file=f)
    print("confusion matrix ", confusion_matrix(Y_test, predictions), file=f)

print(time.time() - prev_time)
