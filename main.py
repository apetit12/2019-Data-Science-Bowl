#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: antoinepetit
"""
# -----------------------------------------------------------------------------
#                               IMPORTING
# -----------------------------------------------------------------------------
import pandas as pd
import json

df = pd.read_csv('Data/train.csv')
df_labels = pd.read_csv('Data/train_labels.csv')
#df_specs = pd.read_csv('Data/specs.csv')
#df_test = pd.read_csv('Data/test.csv')


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

# identify the installation_id that took at least one assessment 
# (some may have not submitted an answer for it)
users = df[df['type']=='Assessment'].installation_id.unique().tolist()

# drop the other installation_ids
train = df[df['installation_id'].isin(users)]
train['new_timestamp'] = pd.to_datetime(train['timestamp'])
train['day'] = train['new_timestamp'].dt.dayofweek
train['time'] = train['new_timestamp'].dt.hour

# identify the assessments (Careful: 4110 is the only end code for Bird measurer, 
# and it is only 4100 for the other assessment types)
assessments = train[(train['type']=='Assessment') & 
                    ((train['event_code']==4110) & (train['title']=='Bird Measurer (Assessment)') 
                    |(train['event_code']==4100) & (train['title']!='Bird Measurer (Assessment)'))]

assessments['Success'] = assessments['event_data'].apply(lambda x: json.loads(x)['correct'])
assessments = pd.get_dummies(assessments, columns=['Success'], dummy_na=False)
assessments.Success_False = assessments.Success_False.astype(int)
assessments.Success_True = assessments.Success_True.astype(int)

# -----------------------------------------------------------------------------
#                               GETTING TARGET VECTOR
# -----------------------------------------------------------------------------

# Construct df_labels
test = assessments.groupby(['game_session','installation_id','title']).agg({'Success_False':'sum','Success_True':'sum'}).reset_index()
test['accuracy'] = test['Success_True'].astype(float)/(test['Success_True']+test['Success_False'])
test['accuracy_group'] = test['accuracy'].apply(lambda x: get_outcome(x),axis=1)

# Select the last assessment for each installation_id and add game title
tmp = assessments[assessments.groupby(['installation_id'])['new_timestamp'].transform(max) == assessments['new_timestamp']]
target = pd.merge(df_labels, tmp, how='right', on=['installation_id','game_session'])[['installation_id','game_session','event_id','title_y','accuracy_group']]

# -----------------------------------------------------------------------------
#                               GETTING PREDICTION VECTOR
# -----------------------------------------------------------------------------

# construct some features for each remaining installation_id
predicted = target[['installation_id','game_session','event_id','title_y']]














