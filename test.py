import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
data = load_dataset("rotten_tomatoes")
data
train_ds = data['train']
train_ds[:5]
df = pd.DataFrame(train_ds)
df
df['label'].value_counts(ascending=False).plot.bar()
plt.title("Frequency of Classes")
plt.show()
model_ckpt = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


data_encoded = data.map(tokenize, batched=True, batch_size=None)
data_encoded['train'].column_names
def extract_hidden_states(batch):
    inputs = {k:v for k,v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
data_encoded.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
data_hidden = data_encoded.map(extract_hidden_states, batched=True)
data_hidden["train"].column_names
x_train = np.array(data_hidden['train']['hidden_state'])
x_test = np.array(data_hidden['test']['hidden_state'])
y_train = np.array(data_hidden['train']['label'])
y_test = np.array(data_hidden['test']['label'])
y_valid = np.array(data_hidden['validation']['label'])
x_valid = np.array(data_hidden['validation']['hidden_state'])


x_train.shape, x_valid.shape, x_test.shape
nb = GaussianNB()
knn = KNeighborsClassifier()
lr = LogisticRegression(max_iter=3000)
svm = SVC()
tree = DecisionTreeClassifier()
mnb = MultinomialNB()
scaler = MinMaxScaler()
m = scaler.fit_transform(x_train)
n = scaler.transform(x_valid)


mnb.fit(m, y_train)
nb.fit(x_train, y_train)
knn.fit(x_train, y_train)
lr.fit(x_train, y_train)
svm.fit(x_train, y_train)
tree.fit(x_train, y_train)


print(f"Multinomial Naive Bayes Accuracy: {mnb.score(n,y_valid)}")
print(f"Gaussian Naive Bayes Accuracy: {nb.score(x_valid,y_valid)}")
print(f"KNN Accuracy: {knn.score(x_valid,y_valid)}")
print(f"Logistic Regression Accuracy: {lr.score(x_valid,y_valid)}")
print(f"SVC Accuracy: {svm.score(x_valid,y_valid)}")
print(f"Decision Tree Accuracy: {tree.score(x_valid,y_valid)}")
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print(f"Random Forest Accuracy: {rf.score(x_valid, y_valid)}")
gx = XGBClassifier()
gx.fit(x_train, y_train)
gx.score(x_valid, y_valid)
logreg = LogisticRegression(C=0.1, l1_ratio=0.2, penalty='elasticnet', solver='saga', max_iter=1000)
logreg.fit(x_train, y_train)
print(f"Best Performance of Logistic Regression: {logreg.score(x_valid, y_valid)}")
svm = SVC(C=100, decision_function_shape='ovo', gamma=0.035)


svm.fit(x_train, y_train)
print(f"Best Performance of SVC: {svm.score(x_valid, y_valid)}")
models = [
    ('xgb', cl),
    ('svc', svm),
    ('logreg', logreg)
]
stacked_model = StackingClassifier(estimators=models, final_estimator=LogisticRegression())


stacked_model.fit(x_train, y_train)


stacked_model.score(x_valid, y_valid)
stacked_model2 = StackingClassifier(estimators=models, final_estimator=SVC())


stacked_model2.fit(x_train, y_train)


stacked_model2.score(x_valid, y_valid)
def performance(model, x_test, y_test):
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print("                 Model Performance")
    print(report)
    print(f"Accuracy = {round(accuracy*100, 2)}%")
    matrix = confusion_matrix(y_test, preds)
    matrix_disp = ConfusionMatrixDisplay(matrix)
    matrix_disp.plot(cmap='Blues')
    plt.show()
performance(rf, x_test, y_test)
performance(gx, x_test, y_test)
performance(svm, x_test, y_test)
performance(logreg, x_test, y_test)
performance(stacked_model2, x_test, y_test)
performance(stacked_model, x_test, y_test)