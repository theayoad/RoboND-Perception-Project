#!/usr/bin/env python
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm, grid_search, cross_validation, metrics
from sklearn import grid_search
from sklearn.preprocessing import LabelEncoder, StandardScaler


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Load training data from disk
training_set = pickle.load(open('training_set.sav', 'rb'))

# Format the features and labels for use with scikit learn
feature_list = []
label_list = []

for item in training_set:
    if np.isnan(item[0]).sum() < 1:
        feature_list.append(item[0])
        label_list.append(item[1])

print('Features in Training Set: {}'.format(len(training_set)))
print('Invalid Features in Training set: {}'.format(len(training_set)-len(feature_list)))


X = np.array(feature_list)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
X_train = X_scaler.transform(X)
y_train = np.array(label_list)

# # Convert label strings to numerical encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)


# Generate classifier
################################################################################
print 'Searching for best classifier in search space...'
start_time = time.time()
parameters = dict(
  kernel=['linear', 'sigmoid'],
  C=[.5, 1, 5, 10, 80, 85, 90, 91, 92, 93, 95, 97, 100],
  gamma=[0.01, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.008, 0.0001, .00015, 0.0007, .0008, .0009, 0])
clf = grid_search.GridSearchCV(svm.SVC(probability=True), parameters, verbose=1).fit(X_train, y_train)
classifier = clf.best_estimator_
print '* Best Parameters (kernel, C, gamma):', clf.best_params_
print '* Time elapsed:', time.time() - start_time

# Create classifier (Uncomment if you want to create classifier  directly
# as opposed to searching for best classifier in search space)
# classifier = svm.SVC(kernel='sigmoid', C=40, gamma=.0001)
################################################################################

# Set up 5-fold cross-validation
kf = cross_validation.KFold(
  len(X_train),
  n_folds=5,
  shuffle=True,
  random_state=1)

# Perform cross-validation
scores = cross_validation.cross_val_score(
  cv=kf,
  estimator=classifier,
  X=X_train, 
  y=y_train,
  scoring='accuracy')
print('Scores: ' + str(scores))
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), 2*scores.std()))

# Gather predictions
predictions = cross_validation.cross_val_predict(
  cv=kf,
  estimator=classifier,
  X=X_train, 
  y=y_train)

accuracy_score = metrics.accuracy_score(y_train, predictions)
print('accuracy score: '+str(accuracy_score))

confusion_matrix = metrics.confusion_matrix(y_train, predictions)
class_names = encoder.classes_.tolist()


# Train the classifier
classifier.fit(X=X_train, y=y_train)

model = {'classifier': classifier, 'classes': encoder.classes_, 'scaler': X_scaler}

# Save classifier to disk
print 'Saving model to disk...'
pickle.dump(model, open('model.sav', 'wb'))
print 'Saved model to disk.'

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=encoder.classes_,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=encoder.classes_, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
