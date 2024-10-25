from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
import warnings


def leave_one_out_for(data, target, classifiers, scoring_metrics):
    loo = LeaveOneOut()
    results = {clf.__class__.__name__: {metric.__name__: None for metric in scoring_metrics} for clf in classifiers}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for clf in classifiers:
            print(clf.__class__.__name__)
            y_true, y_pred = [], []
            for train_idx, test_idx in tqdm(loo.split(data), total=data.shape[0]):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]
                clf.fit(X_train, y_train)
                y_pred.append(clf.predict(X_test)[0])
                y_true.append(y_test[0])   

            for metric in scoring_metrics:
                score = metric(y_true, y_pred)
                results[clf.__class__.__name__][metric.__name__] = round(score, 5)
        
    return results

if __name__ == "__main__":
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

    classifiers = [RandomForestClassifier(), LogisticRegression(max_iter=250)]
    metrics = [accuracy_score, f1_score, recall_score]

    result = leave_one_out_for(X, y, classifiers, metrics)
    print(result)


