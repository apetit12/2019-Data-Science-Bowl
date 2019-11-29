#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
11/28/2019

@author: antoinepetit

This is the first submission for the competition.

- Model: use the most frequent group for each assessment type as the predicted value
- Training set score: 0.2547
- Public score: 0.395

"""
# -----------------------------------------------------------------------------
#                               IMPORTING
# -----------------------------------------------------------------------------
import pandas as pd
import json
from sklearn.metrics import cohen_kappa_score

print('Reading train.csv file....')
df_train = pd.read_csv('Data/train.csv')
print('Reading train_labels.csv file....')
df_labels = pd.read_csv('Data/train_labels.csv')
#df_specs = pd.read_csv('Data/specs.csv')
print('Reading test.csv file....')
df_test = pd.read_csv('Data/test.csv')

# -----------------------------------------------------------------------------
#                               FUNCTIONS
# -----------------------------------------------------------------------------
def get_outcome(val):
    '''
    Map the accuracy with the outcome group
    '''
    if val==1.0:
        return 3
    elif val==0.5:
        return 2
    elif val>0.0:
        return 1
    else:
        return 0

# -----------------------------------------------------------------------------
#                               PREPROCESSING
# -----------------------------------------------------------------------------
print('Starting preprocessing....')

# identify the installation_id that took at least one assessment 
# (some may have not submitted an answer for it)
users_train = df_train[df_train['type']=='Assessment'].installation_id.unique().tolist()

# drop the other installation_ids
df_train = df_train[df_train['installation_id'].isin(users_train)]
df_train.loc[:,'new_timestamp'] = pd.to_datetime(df_train['timestamp'])
df_test.loc[:,'new_timestamp'] = pd.to_datetime(df_test['timestamp'])

# identify the assessments (Careful: 4110 is the only end code for Bird measurer, 
# and it is only 4100 for the other assessment types)
assessments = df_train[(df_train['type']=='Assessment') & 
                    ((df_train['event_code']==4110)&(df_train['title']=='Bird Measurer (Assessment)') 
                    |(df_train['event_code']==4100)&(df_train['title']!='Bird Measurer (Assessment)'))]

assessments.loc[:,'Success'] = assessments['event_data'].apply(lambda x: json.loads(x)['correct'])
assessments = pd.get_dummies(assessments, columns=['Success'], dummy_na=False)
assessments.loc[:,'Success_False'] = assessments.Success_False.astype(int)
assessments.loc[:,'Success_True'] = assessments.Success_True.astype(int)

# -----------------------------------------------------------------------------
#                  GETTING TARGET VECTOR FOR TRAINING SET
# -----------------------------------------------------------------------------
print('Getting target vector....')

# Construct df_labels
train_labels = assessments.groupby(['game_session','installation_id','title']).agg({
                                                                        'Success_False':'sum',
                                                                        'Success_True':'sum'}).reset_index()
train_labels['accuracy'] = train_labels['Success_True'].astype(float)/(train_labels['Success_True']+train_labels['Success_False'])
train_labels['accuracy_group'] = train_labels['accuracy'].apply(lambda x: get_outcome(x))
train_labels = pd.get_dummies(train_labels, columns=['accuracy_group'], dummy_na=False)
train_labels.rename(columns={'accuracy_group_0':'0', 'accuracy_group_1':'1',
                     'accuracy_group_2':'2', 'accuracy_group_3':'3'},inplace=True)

# Select the last assessment for each installation_id and add game title
last_assessment_train = assessments[assessments.groupby(['installation_id'])['new_timestamp'].transform(max)==assessments['new_timestamp']]   
target = pd.merge(df_labels, last_assessment_train, how='right', on=['installation_id','game_session'])[['installation_id',
                                                                                     'game_session',
                                                                                     'event_id',
                                                                                     'title_y',
                                                                                     'accuracy_group']]

# -----------------------------------------------------------------------------
#                 GETTING PREDICTION VECTOR FOR TRAINING SET
# -----------------------------------------------------------------------------
print('Compute prediction....')

# Obtain most likely category for each assessment title
groups = train_labels.groupby(['title']).agg({'0':'sum','1':'sum','2':'sum','3':'sum'})
groups['accuracy_group'] = groups.idxmax(axis=1)
groups.reset_index(inplace=True)
groups.rename(columns={'title':'title_y'},inplace=True)

# Predict outcome category based on most frequent one
predicted = target[['installation_id','game_session','event_id','title_y']]
predicted = pd.merge(predicted, groups, how='left', on=['title_y'])
predicted.accuracy_group = predicted.accuracy_group.astype(int)

# -----------------------------------------------------------------------------
#              COMPUTE WEIGHTED KAPPA METRIC ON TRAINING SET
# -----------------------------------------------------------------------------
print('... And kappa metric for training set....')

results_training = pd.merge(predicted,target,how='inner', on=['installation_id'])
print(cohen_kappa_score(results_training['accuracy_group_x'],results_training['accuracy_group_y']))

# -----------------------------------------------------------------------------
#                 GETTING PREDICTION VECTOR FOR TESTING SET
# -----------------------------------------------------------------------------
print('Compute prediction for testing set....')

last_assessment_test = df_test[df_test.groupby(['installation_id'])['new_timestamp'].transform(max)==df_test['new_timestamp']]
groups.rename(columns={'title_y':'title'},inplace=True)
results_testing = pd.merge(last_assessment_test, groups, how='left', on=['title'])[['installation_id','accuracy_group']]

results_testing.to_csv('submission.csv', index = False)
