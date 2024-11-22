from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def k_fold_cross_validation(data, target, classifiers, scoring_metrics, k=10):
    results = {clf.__class__.__name__: {metric: {} for metric in scoring_metrics} for clf in classifiers}

    for clf in tqdm(classifiers):
        cv = ShuffleSplit(n_splits=k)
        scores = cross_validate(clf, data, target, cv=cv, scoring=scoring_metrics)
        for metric in scoring_metrics:
            mean_score = sum(scores['test_' + metric]) / len(scores['test_' + metric])
            max_score = max(scores['test_' + metric])
            min_score = min(scores['test_' + metric])
            results[clf.__class__.__name__][metric]['avg'] = round(mean_score, 5)
            results[clf.__class__.__name__][metric]['max'] = round(max_score, 5)
            results[clf.__class__.__name__][metric]['min'] = round(min_score, 5)


    return results

if __name__ == "__main__":
    
    X, y = make_classification(n_samples=250, n_features=5, n_classes=2, random_state=42)

    classifiers = [
        RandomForestClassifier(), 
        LogisticRegression(max_iter=250), 
        KNeighborsClassifier(n_neighbors=3)
    ]

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'matthews_corrcoef'] 

    result = k_fold_cross_validation(X, y, classifiers, metrics)

    print(result)


