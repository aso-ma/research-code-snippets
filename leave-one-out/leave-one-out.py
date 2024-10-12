from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
import warnings


def leave_one_out_for(data, target, classifiers, scoring_metrics):
    loo = LeaveOneOut()
    results = {clf.__class__.__name__: {metric.__name__: [] for metric in scoring_metrics} for clf in classifiers}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for clf in classifiers:
            print(clf.__class__.__name__)
            for train_idx, test_idx in tqdm(loo.split(data), total=data.shape[0]):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                for metric in scoring_metrics:
                    score = metric(y_test, y_pred)
                    results[clf.__class__.__name__][metric.__name__].append(score)

    avg_results = {}
    for clf_name, metrics in results.items():
        avg_results[clf_name] = {metric: sum(scores)/len(scores) for metric, scores in metrics.items()}
    return avg_results

if __name__ == "__main__":
    df_iris = datasets.load_iris()
    X = df_iris.data  
    y = df_iris.target 

    classifiers = [RandomForestClassifier(), LogisticRegression(max_iter=250)]
    metrics = [accuracy_score, f1_score, recall_score]

    result = leave_one_out_for(X, y, classifiers, metrics)

    print(result)


