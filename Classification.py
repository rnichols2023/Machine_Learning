#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import sklearn
assert sys.version_info >= (3, 7)


# In[2]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", as_frame = False)


# In[3]:


X, y = mnist.data, mnist.target


# In[4]:


print(mnist.DESCR)


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image, cmap = "binary")
    plt.axis("off")


# In[7]:


plot_digit(X [5])
plt.show


# In[8]:


from sklearn.linear_model import SGDClassifier


# In[9]:


y_train_3 = (y[:60000] == '3')
y_test_3 = (y[60000:] == '3')


# In[10]:


sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X[:60000], y_train_3)


# In[11]:


sgd_clf.predict([X[6]])


# In[12]:


from sklearn.model_selection import cross_val_score


# In[13]:


cross_val_score(sgd_clf, X[:60000], y_train_3, cv = 3, scoring = "accuracy")


# In[14]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# In[15]:


y_train_pred = cross_val_predict(sgd_clf, X[:60000], y_train_3, cv = 3,)
confusion_matrix(y_train_3, y_train_pred)


# In[16]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[17]:


Precision = precision_score(y_train_3, y_train_pred)
Recall = recall_score(y_train_3, y_train_pred)
F1 = f1_score(y_train_3, y_train_pred)
print(f"Precision: {Precision:.2f}, Recall: {Recall:.2f}, F1-score: {F1:.2f}")


# In[18]:


from sklearn.metrics import roc_curve
import numpy as np


# In[19]:


y_scores = cross_val_predict(sgd_clf, X[:60000], y_train_3, cv = 3, method = "decision_function")
fpr, tpr, thresholds = roc_curve(y_train_3, y_scores)


# In[29]:


plt.plot(fpr, tpr, linewidth = 2, label = "SGDClassifier")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("roc_curve.png", dpi = 300, bbox_inches = "tight")
plt.show()


# In[21]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_train_3, y_train_pred)
precision = precision_score(y_train_3, y_train_pred)
recall = recall_score(y_train_3, y_train_pred)
f1 = f1_score(y_train_3, y_train_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[22]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_train_3, y_scores)
print("AUC Score:", auc)


# In[23]:


y_scores = cross_val_predict(sgd_clf, X[:60000], y_train_3, cv=3, method="decision_function")


# In[24]:


from sklearn.metrics import precision_score, recall_score

threshold_high = 1000
y_pred_high = (y_scores > threshold_high)

print("High Threshold (1000)")
print("Precision:", precision_score(y_train_3, y_pred_high))
print("Recall:", recall_score(y_train_3, y_pred_high))


# In[25]:


threshold_low = -1000
y_pred_low = (y_scores > threshold_low)

print("Low Threshold (-1000)")
print("Precision:", precision_score(y_train_3, y_pred_low))
print("Recall:", recall_score(y_train_3, y_pred_low))


# In[26]:


thresholds = [-2000, -1000, 0, 1000, 2000]

for t in thresholds:
    preds = (y_scores > t)
    p = precision_score(y_train_3, preds)
    r = recall_score(y_train_3, preds)
    print(f"Threshold {t:>5}: Precision={p:.3f}, Recall={r:.3f}")


# In[ ]:




