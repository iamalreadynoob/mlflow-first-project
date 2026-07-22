import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# set the experiment id
mlflow.set_experiment(experiment_id="sado-test")

mlflow.autolog()
df = pd.read_csv("Student_performance_data _.csv")

y = df["GradeClass"].values
del df["GradeClass"]
X = df.values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=True
)

rf = RandomForestClassifier(
    n_estimators=20,
    max_depth=10
)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)