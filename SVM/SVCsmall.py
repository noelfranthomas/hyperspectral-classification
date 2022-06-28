import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from joblib import dump, load
import glob

# Try LIBSVM next for GPU support
# Or, https://github.com/murtazajafferji/svm-gpu

root = '/home/noelt/scripts/SVM/data/'
train_f = 'train_data.csv'

train_data = pd.read_csv(root + train_f) # Load training data

train_data.drop(columns='intensity', inplace=True) # Drop intensity

_head = train_data.head(250000) # Reduce size (N = 250,000, t = 3505s)
_tail = train_data.tail(250000)
_frames = [_head, _tail]
train_data = pd.concat(_frames)

# Fix class imbalance - Upsampling
def Upsample():
    majority = df[df.balance==0]
    minority = df[df.balance==1]

# Fix class imbalance - Downsampling
def Downsample():
    nMono_trainSet = np.count_nonzero(trainSet == 'mono')
    nFibr_trainSet = np.count_nonzero(trainSet == 'fibr')
    print("nMono in training set: %d" % nMono_trainSet)
    print("nFibr in training set: %d" % nFibr_trainSet)

    mono_trainSubset = trainSet[trainSet['ID'] == 'mono']   # extract all the mono rows
    fibr_trainSubset = trainSet[trainSet['ID'] == 'fibr']   # extract all the fibr rows

    if nMono_trainSet < nFibr_trainSet:
        fibr_trainSubset = fibr_trainSubset.sample(nMono_trainSet)   # extract the same # of fibr rows as we have mono rows
    elif nMono_trainSet > nFibr_trainSet:
        mono_trainSubset = mono_trainSubset.sample(nFibr_trainSet)   # extract the same # of mono rows as we have fibr rows

    # now combine the 2 subsets into a single training subset
    frames = [mono_trainSubset,fibr_trainSubset]
    trainSubset = pd.concat(frames)   # this will be the main training set

    # let's check
    nMono_trainSubet = np.count_nonzero(trainSubset == 'mono')
    nFibr_trainSubet = np.count_nonzero(trainSubset == 'fibr')
    print("nMono in training subset: %d" % nMono_trainSubet)
    print("nFibr in training subset: %d" % nFibr_trainSubet)

# Upsample 
# ...
# Downsample
# ...



X_train = train_data.drop(['ID'], axis=1)
y_train = np.squeeze(train_data['ID'])

print('Entered training array\n')

# # Upsample, Scaled
# print('Training upsample, scaled')
# us_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf', verbose=1))
# us_model.fit(X_train, y_train)
# us_fSave = root+'us_model.joblib'
# dump(us_model,us_fSave)
# print('\n')

# # Upsample, No Scaling
# print('Training upsample, no scaling')
# u_model = SVC(gamma='auto', kernel='rbf', verbose=1)
# u_model.fit(X_train, y_train)
# u_fSave = root+'u_model.joblib'
# dump(u_model,u_fSave)
# print('\n')

# # Downsample, Scaled
# print('Training downsample, scaled')
# ds_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf', verbose=1))
# ds_model.fit(X_train, y_train)
# ds_fSave = root+'ds_model.joblib'
# dump(ds_model,ds_fSave)
# print('\n')

# Downsample, No Scaling
print('Training downsample, no scaling')
d_model = SVC(gamma='auto', kernel='rbf', verbose=1)
d_model.fit(X_train, y_train)
d_fSave = root+'d_model.joblib'
dump(d_model,d_fSave)
print('\n')

test_folder = root + 'test/*'
test_array = []

for f in glob.glob(test_folder):
  print('Loading: ' + f)
  temp = pd.read_csv(f)
  temp.replace('mono.*', int(0) ,regex=True, inplace = True)
  temp.replace('fibr.*', int(1) ,regex=True, inplace = True)
  temp['ID'] = temp['ID'].map({0 : 'mono', 1 : 'fibr'}).astype(str)
  temp.drop(['intensity'], axis=1, inplace=True)
  test_array.append(temp)

accuracy_array = []

for test in test_array:
  x_test = test.drop(['ID'], axis=1)
  y_test = test.drop(['G1', 'G2', 'S1', 'S2'], axis=1)
  y_predicted_test = d_model.predict(x_test)
  score = accuracy_score(y_test, y_predicted_test)
  accuracy_array.append(score)

x = np.array(accuracy_array)
print(np.unique(x))

print(np.sum(x)/len(x))