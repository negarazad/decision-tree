from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# بارگذاری مجموعه داده‌ای Iris
iris = load_iris()
X, y = iris.data, iris.target

# ساخت مدل درخت تصمیم‌گیری
clf = DecisionTreeClassifier()
clf.fit(X, y)

# ارزیابی دقت قبل از هرس
scores_before = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy before pruning: {scores_before.mean()}")

# ساخت مدل درخت تصمیم‌گیری با هرس پس‌گیرانه
clf_pruned = DecisionTreeClassifier(max_depth=3)
clf_pruned.fit(X, y)

# ارزیابی دقت بعد از هرس
scores_after = cross_val_score(clf_pruned, X, y, cv=5)
print(f"Accuracy after pruning: {scores_after.mean()}")

# رسم درخت تصمیم‌گیری قبل و بعد از هرس
plt.figure(figsize=(12, 8))
plt.subplot(121)
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Before Pruning")

plt.subplot(122)
plot_tree(clf_pruned, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("After Pruning")

plt.show()
