import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine=load_wine()
x=wine.data
y=wine.target

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.1,random_state=42)

max_depth=10
n_estimators=5

mlflow.autolog()
mlflow.set_experiment("wine_prediction")

with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X_train,Y_train)
    
    Y_pred=rf.predict(X_test)
    accuracy=accuracy_score(Y_pred,Y_test)
    
    cm=confusion_matrix(Y_pred,Y_test)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix (Wine classification)')
    
    plt.savefig('cm.png')
    
    mlflow.log_artifact(__file__)
    mlflow.set_tags({"Author":"VISHNU TEJA","Project":"wine classification"})
    
    print(accuracy)
    