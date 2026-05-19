from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()

X=wine.data
y=wine.target

# 8:2 분할
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

print(X_train.shape)
print(X_test.shape)