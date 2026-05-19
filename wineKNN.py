from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

wine = datasets.load_wine()

X=wine.data
y=wine.target

# 8:2 분할
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
scores = metrics.accuracy_score(y_test,y_pred)
print(scores)