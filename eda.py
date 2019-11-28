#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: antoinepetit
"""
# -----------------------------------------------------------------------------
#                               IMPORTING
# -----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

df = pd.read_csv('Data/train.csv')
df_labels = pd.read_csv('Data/train_labels.csv')
print(len(df_labels['installation_id'].unique().tolist()))
#df_specs = pd.read_csv('Data/specs.csv')
#df_test = pd.read_csv('Data/test.csv')
df.head(10)

# -----------------------------------------------------------------------------
#                               PREPROCESSING
# -----------------------------------------------------------------------------

# identify the installation_id that took at least one assessment 
# (some may have not submitted an answer for it)
users = df[df['type']=='Assessment'].installation_id.unique().tolist()
print(len(users))

# drop the other installation_ids
train = df[df['installation_id'].isin(users)]
train['new_timestamp'] = pd.to_datetime(train['timestamp'])

# extracting date from timestamp
train['day'] = train['new_timestamp'].dt.dayofweek

# extracting time from timestamp
train['time'] = train['new_timestamp'].dt.hour

# identify the assessments (Careful: 4110 is the only end code for Bird measurer, 
# and it is only 4100 for the other assessment types)
assessments = train[(train['type']=='Assessment') & 
                    ((train['event_code']==4110) & (train['title']=='Bird Measurer (Assessment)') 
                    |(train['event_code']==4100) & (train['title']!='Bird Measurer (Assessment)'))]

assessments['Success'] = assessments['event_data'].apply(lambda x: json.loads(x)['correct'])

# -----------------------------------------------------------------------------
#                               PLOTTING
# -----------------------------------------------------------------------------

# Plot number of users taking assessmentss
fig0 = plt.figure(0)
ax0 = fig0.add_subplot(111)
plt.bar([1,2,3],
        [len(df['installation_id'].unique().tolist()),
           len(train['installation_id'].unique().tolist()),
           len(assessments['installation_id'].unique().tolist())])
plt.title('Funnel analysis of users taking assessments')
plt.ylabel('Count')
plt.xticks([1,2,3],['Total users','Users who started an assessment','Users who completed an assessment'])

# Plot Success/Failure for each assessment category
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
sns.countplot(x="title", hue='Success', data=assessments)
plt.title('Distribution of success across all assessment category')
plt.xlabel('')

# Plot activity by day
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
plt.plot(train.groupby(pd.Grouper(key='new_timestamp', freq='D'))['event_id'].count())
plt.ylabel('Events')
plt.xlabel('Date')
plt.title('Number of game events per day')

# Plot activity by day of week
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
plt.plot(train.groupby(['day'])['event_id'].count())
plt.ylabel('Events')
plt.xlabel('Weekday')
plt.xticks([0,1,2,3,4,5,6],['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title('Number of game events per weekday')

# Plot activity by time of day
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
plt.plot(train.groupby(['time'])['event_id'].count())
plt.ylabel('Events')
plt.xlabel('Hour')
plt.title('Number of game events per time of day')

# Plot activity by installation_id
temp=train.groupby(['installation_id'])['event_id'].count().reset_index().sort_values(by=['event_id'],ascending=False)
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)
sns.boxplot(y='event_id', data=temp)
plt.ylabel('Events')

# Plot game_time by session
temp = train.groupby(['game_session']).agg({'installation_id':'count','event_id':'count','game_time':'max'})
fig6 = plt.figure(6)
ax6 = fig6.add_subplot(111)
sns.distplot(temp['game_time'].apply(np.log1p).values,bins=100)

# Plot Success/Failure for each assessment category
fig7 = plt.figure(7)
ax7 = fig7.add_subplot(111)
sns.countplot(x="type", data=train)
plt.title('Distribution of event categories')
plt.xlabel('')

