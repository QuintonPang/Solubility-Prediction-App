import pandas as pd

# obtaining dataet
delaney_solubility_with_descriptors_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'
dataset = pd.read_csv(delaney_solubility_with_descriptors_url)

# drop column 'logS'
X = dataset.drop(['logS'],axis=1)

# first parameter is row, second parameter is column
# in this case it means all rows and last column which is 'logS'
Y = dataset.iloc[:,-1]

from sklearn import linear_model

# build model
model = linear_model.LinearRegression()
model.fit(X,Y)

# export model
import pickle
pickle.dump(model,open('../solubility-model.pkl','wb'))
