from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from scipy.stats.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime
from dateutil.parser import parse
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

#load main data
main = []
for i in range(7):
    file = 'main/main201{0}.csv'.format(str(i+1))
    main.append(pd.read_csv(file, index_col = 0))

# load skill data, do some data transformation, convert month to year_month
skill = []
for i in range(7):
    file = 'skill/skill201{0}.csv'.format(str(i+1))
    skill.append(pd.read_csv(file, index_col = 0).set_index(['gvkey','month','ID']).unstack('ID').fillna(0))
for i in range(7):
    skill[i] = skill[i].reset_index()
    skill[i]['month'] = (str(2011+i) + skill[i]['month'].map(str)).map(int)
    skill[i].rename(columns={'month':'date'}, inplace = True)
    skill[i] = skill[i].set_index(['gvkey', 'date'])

# get return data from main data
return_total = pd.DataFrame()
for i in range(7):
    return_total = pd.concat([return_total, main[i][['gvkey','date','return']]])

# concat skill data together
skill_total = skill[0]
for i in np.arange(1,7):
    skill_total = pd.concat([skill_total, skill[i]], join = 'inner')

# function for transfering string to timestamp
def str_to_datetime(string):
    return datetime.strptime(string, '%Y%m')

# change return value to label for classification
def change_label(dataframe):
    dataframe_sorted = dataframe.sort_values('return', ascending = False)
    dataframe_sorted_good = dataframe_sorted.iloc[:int(dataframe_sorted.shape[0]/10)]
    dataframe_sorted_bad = dataframe_sorted.iloc[int(dataframe_sorted.shape[0]*(9/10)):]
    dataframe_sorted_good['return'] = 1
    dataframe_sorted_bad['return'] = -1
    dataframe_combined = pd.concat([dataframe_sorted_good, dataframe_sorted_bad])
    return dataframe_combined

# function for creating factors and return using classification.
def classification_create_factors(df, model):
    FACTOR = pd.DataFrame()
    RETURN = pd.DataFrame()
    time = pd.date_range('2011-01-01', '2017-12-01', freq='MS')
    length = len(time)
    for i in range(length-1):
        time1 = df[df['date']==time[i]]
        time2 = df[df['date']==time[i+1]]
        train = change_label(time1)
        scaler = preprocessing.StandardScaler().fit(train.iloc[:,2:-1])
        train_X = scaler.transform(train.iloc[:,2:-1])
        test_X = scaler.transform(time2.iloc[:,2:-1])
        clf = GradientBoostingClassifier()
        clf = clf.fit(train_X, train['return'])
        model_feature = SelectFromModel(clf, prefit=True)
        model.fit(model_feature.transform(train_X), train['return'])
        prediction = model.predict_proba(model_feature.transform(test_X))
        factor = pd.concat([time2[['gvkey','date']].reset_index(drop=True), pd.DataFrame({'factor':prediction[:,1]})], axis = 1)
        return_ = time2[['gvkey','date','return']]
        FACTOR = pd.concat([FACTOR, factor])
        RETURN = pd.concat([RETURN, return_])
        print(time[i])
    FACTOR = FACTOR.set_index(['date','gvkey'])
    FACTOR = FACTOR.unstack('date')
    RETURN = RETURN.set_index(['date','gvkey'])
    RETURN = RETURN.unstack('date')
    return FACTOR, RETURN

# function for creating factors using regression.
def regression_create_factors(df, model):
    FACTOR = pd.DataFrame()
    time = pd.date_range('2011-01-01', '2017-12-01', freq='MS')
    length = len(time)
    for i in range(length-1):
        time1 = df[df['date']==time[i]]
        time2 = df[df['date']==time[i+1]]
        model.fit(time1.iloc[:,2:-1], time1['return'])
        prediction = model.predict(time2.iloc[:,2:-1])
        factor = pd.concat([time2[['gvkey','date']].reset_index(drop=True), pd.DataFrame({'factor':prediction})], axis = 1)
        FACTOR = pd.concat([FACTOR, factor])
        print(i)
    FACTOR = FACTOR.set_index(['date','gvkey'])
    FACTOR = FACTOR.unstack('date')
    return FACTOR

# create skill_return dataframe
X_new = skill_total
skill_total_1 = pd.DataFrame(X_new, index = skill_total.index)
data_set_1 = pd.merge(skill_total_1, return_total.set_index(['gvkey','date']), left_index=True, right_index=True, how='inner')
data_set_1 = data_set_1.reset_index()
data_set_1['date'] = (data_set_1['date'].map(str)).map(str_to_datetime)

# create factors by using GradientBoostingClassifier without GridSearchCV.
GB = GradientBoostingClassifier()
GB_factor, Return = classification_create_factors(data_set_1, GB)

# create factors by using GradientBoostingClassifier with GridSearchCV.
parameters = {'learning_rate':[0.1,0.5], 'n_estimators':[100,50],'max_depth':[3,2]}
GB = GradientBoostingClassifier()
clf = GridSearchCV(GB, parameters, cv=3)
GB_GS_factor, Return = classification_create_factors(data_set_1, clf)

# save factors and return as csv file
GB_factor['factor'].to_csv('Factor_Return/GB_factor.csv')
GB_GS_factor['factor'].to_csv('Factor_Return/GB_GS_factor.csv')
Return['return'].to_csv('Factor_Return/Return.csv')
