#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:37:48 2020

@author: christopher
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
from google.cloud import bigquery
import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import auc,confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
colors = ['#42ffb7', '#00ffff', '#828cfb', '#0068e8', '#e06fa9']
sns.set_palette(sns.color_palette(colors))


# Exporation

# Query from GCS, return df
def query (JSON_File, Project):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ={JSON_File}
    bigquery_client = bigquery.Client(project={Project})
    QUERY = """
    SELECT 
        * 
    FROM 
        `events` 
    WHERE 
        DATE(event_timestamp) = "2020-03-02" 
    LIMIT 
        1000
    """
    query_job = bigquery_client.query(QUERY)
    df = query_job.to_dataframe()
    # Save query
    df.to_csv('df.csv', index=False)
    return df

def drop_duplicates(df):
    df = df.drop_duplicates()
    return df
# Change date to datetime format
def date_time(df):
    df.ds = pd.to_datetime(df.ds)
    return df

# DAU graph _ exploration graph
def DAU_graph(df):
    plt.figure(figsize=(40,8))
    ax = sns.lineplot(y="avg_daily_sessions_duration", x='ds', data=df)
    ax.set(xlabel='Date', ylabel='Average Daily Session in Minuets')
    ax.title('Average Daily Sessions')
    ax.savefig('Average Daily Sessions', dpi=600)
    return df

# Create weekday and month column
def week_month_cols(df):
    df['weekday']=df.ds.dt.day_name()
    df['month'] = df.ds.dt.month_name()
    return df

# Average duration/day
#df_day_duration = df[['player_id', 'duration', 'ds']].groupby('ds').agg('mean')

# Total playtime /playerID
#df_total_duration = df_total_playtime[['player_id', 'total_playtime']].groupby('player_id').agg('mean')

# Add avg duration to df
def avg_session_col(df):
    #Average duration per player
    df_avg_player_session = df[['player_id', 'duration']].groupby('player_id').agg(lambda x: x.unique().sum()/x.nunique())
    # Reset index
    df_avg_player_session = df_avg_player_session.reset_index()
    #create dic with playerID and duration
    dct = {}
    for i, j in zip(list(df_avg_player_session['player_id']), list(df_avg_player_session['duration'])):
        dct[i]=j
    # Map dict to new column
    df['avg_session']=df['player_id'].map(dct)
    return df

def avg_player_session(df):
    current_avg_player_session = df.describe().duration[1]
    df['avg_category']=df['avg_session'].apply(lambda x: 1 if x < current_avg_player_session else 0)
    return df

# Total playtime query + merge to df
def total_playetime_query(df):
    QUERY = """ SELECT * FROM {player statistics} """
    bigquery_client = bigquery.Client(project={Project})
    query_job = bigquery_client.query(QUERY)
    df_total_playtime = query_job.to_dataframe()
    df_total_playtime.to_csv('total_playtime.csv', index=False)
    df = pd.merge(df, df_total_playtime, on='player_id')
    return df

# Drop players with no playtime
def clean_cols(df):    
    df = df.drop(df.index[df['avg_playtime'] == 0])
    df['avg_category'] = df['avg_playtime'].apply(lambda x: 1 if x < current_avg_player_session else 0)
    return df

# Number of days played, events per player, avg playtime, and days played
def player_stats(df):
    # Days played
    player_days_played = pd.pivot_table(data=df,index=('player_id','ds'),values='duration').groupby('player_id').agg('count')
    player_days_played = player_days_played.rename(columns={'duration':'days_played'})
    df = pd.merge(df, player_days_played, on='player_id')
    # Frequency of top 7 Actions plot
    plt.figure(figsize=(10,8))
    action_plot = sns.barplot(x='unique_values', y='counts', data=df4_values.head(7))
    action_plot.set(yticklabels=[])
    action_plot.set(xlabel='Player Actions', ylabel='Frequency')
    plt.title('Frequency of Player Actions')
    plt.savefig('Frequency of Player Actions', dpi=600)
    # Number of events per player
    df_analyzed=pd.merge(df_analyzed, df[['player_id', 'event_class']].query('event_class == "weapon"').groupby('player_id').agg('count'), on='player_id')
    #Event class
    event_class_list = df.event_class.unique().tolist()
    for i in event_class_list:
        df = pd.merge(df, df.query(f'event_class == "{i}"').groupby('player_id').event_class.count(),
             on='player_id', how='outer').fillna(0)
        df.rename(columns={'event_class': f"{i}"}, inplace=True)
    event_type_list = df.event_type.unique().tolist()
    #Event type
    for i in event_type_list:
        df = pd.merge(df, df.query(f'event_class == "{i}"').groupby('player_id').event_class.count(),
             on='player_id', how='outer').fillna(0)
        df.rename(columns={'event_class': f"{i}"}, inplace=True)
    # Avg playtime  
    y = df_player_sessions[['player_id', 'duration']].groupby('player_id').agg('mean')
    y= y.reset_index()
    df = pd.merge(df, y, on='player_id', how='outer').fillna(0)
    df = df.rename(columns={'duration':'avg_playtime'})
    # Days played
    z  = df_player_sessions[['player_id', 'ds']].groupby('player_id').agg(len)
    z = z.reset_index()
    df  = pd.merge(df6, z, on='player_id', how='outer').fillna(0)
    df  = df.rename(columns={'ds':'days_played'})
    return df

#Player Demopgrahics
def players_df():
    # Read csv
    df_players = pd.read_csv('Playtesters_PreAlpha - Playtesters.csv')
    #Clean cols
    df_players.columns = df_players.iloc[0]
    df_players = df_players.drop(0, axis=0)
    # Drop DareWise players
    df_players = df_players.drop(df_players.query('Status == "Darewise"').index)
    # Drop unneeded cols
    cols = [3,4,8,13,15,16,17,18,22,23,24]
    df_players.drop(df_players.columns[cols],axis=1, inplace=True)
    df_players.drop(columns=['Start Date (UTC)', 'Submit Date (UTC)', '#', 'Network ID', 'Status'], inplace=True)
    #Rename columns
    df_players.rename(columns={'Are you currently living in the United States or Canada?': 'us_canada',
                              'Which gender do you identify as?': 'gender',
                              'What is your preferred genre of video games?': 'genre',
                              "Does your computer meet or exceed the minimum system requirements (HD 1080p screen resolution with the default game video options)? For 30 FPS:CPU: Intel i7-4770 @3.5GHz or AMD Ryzen 5 @3.2GHzGPU: Nvidia GTX 1070Ti or AMD Vega 56RAM: 8GBOS: Windows 10": 'pc',
                              """Are you currently involved in an online gaming community? (Online gaming communities include but are not limited to subreddits about a specific game or several games, Discord servers, online web forums and Facebook pages.) """: 'community',
                              "Pick the choice that best describes how you feel about PvP (player-versus-player) play in online games.": 'pvp',
                              "What is your preferred gaming platform?": 'platform',
                              'How old are you?': 'age',
                              'What is your email address?': 'email',
                               'World of Warcraft': 'WoW',
                               'ARK: Survival Evolved': 'Ark',
                              'Community Champion': 'champion'}, inplace=True)
    df_players = df_players.loc[df_players['age'].isna() == False]
    return df_players

def clean_players_df(df_players):
    # Clean champion
    champion_dict = {'FALSE':0, 'TRUE':1}
    df_players.champion = df_players.champion.map(champion_dict)
    df_players.champion.fillna('Other', inplace=True)
    # Clean wave, status
    df_players.us_canada.fillna('Unknown', inplace=True)
    df_players.email.fillna('Unknown', inplace=True)
    df_players.gender.fillna('Unknown', inplace=True)
    # Clean platform
    platform_dict = {'PC':'PC', 'Console':'Console', 'Mobile': 'Mobile', 'Multiplatform - I play how I want based on when I want': 'Multiplatform'}
    df_players.platform = df_players.platform.map(platform_dict)
    df_players.platform.fillna('Other', inplace=True)
    # Clean genre
    genre_dict = {'Massively Multiplayer Online Role Playing Games (MMORPG)':'MMORPG',
                     'Sandbox':'Sandbox', 'Action adventure': 'Action adventure', 
                     'First person shooter (FPS)': 'FPS',
                    'Strategy': 'Strategy', 'Simulation': 'Simulation'}
    df_players.genre = df_players.genre.map(genre_dict)
    df_players.genre.fillna('Other', inplace=True)
    # Clean community
    community_dict = {"Yes - I'm an active part of an existing gaming community and frequently communicate and play with other community members":'Medium Activity',
                     "No - I'm not involved or part of any online gaming communities.":'Not Involved',
                  "Yes - I'm part of a gaming community but not that active.": 'Low Activity', 
                     "Yes - I am or have been a moderator.": 'High Activity',
                    "Yes - I'm part of many gaming communities but not that active in any of them.": 'No Activity'}
    df_players.community = df_players.community.map(community_dict)
    df_players.community.dropna(inplace=True)
    # clean WoW
    WoW_dict = {'World of Warcraft': 1}
    df_players.WoW = df_players.WoW.map(WoW_dict)
    df_players.WoW.fillna(0, inplace=True)
    # Clean ark
    Ark_dict = {'ARK: Survival Evolved': 1}
    df_players.Ark = df_players.Ark.map(Ark_dict)
    df_players.Ark.fillna(0, inplace=True)
    # Clean fortnite
    Fortnite_dict = {'Fortnite': 1}
    df_players.Fortnite = df_players.Fortnite.map(Fortnite_dict)
    df_players.Fortnite.fillna(0, inplace=True)
    # Clean pvp
    pvp_dict = {'Competition is fun.': 'Love',
               "PvP is OK but not the main reason I play games.": 'Ok',
               "I'm here to crush my enemies and see them suffer!": "Love",
               "I don't like PvP and avoid it like the plague.": 'Dislike'}
    df_players.pvp = df_players.pvp.map(pvp_dict)
    df_players.pvp.dropna(inplace=True)
    #Clean Wave
    df_players.Wave.fillna('Unknown', inplace=True)
    wave_dict = {'Unknown': 'Not a Player',
               "1": 'Player',"2": 'Player',"3": 'Player',"4": 'Player',
                 "5": 'Player',"6": 'Player',"7": 'Player',"8": 'Player'}
    df_players.Wave = df_players.Wave.map(wave_dict)
    #Drop NaN values
    df_players.dropna(inplace=True)
    # Export alpha testers to csv
    df_players.to_csv('Alpha_testers_model.csv', index=False)
    return df_players

def merge_alphatesters(df_players):
    # Find all current active players
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ={api}
    bigquery_client = bigquery.Client(project={project})
    QUERY = """SELECT * FROM {[playertesters]}"""
    query_job = bigquery_client.query(QUERY)
    df_testers = query_job.to_dataframe()
    df_testers.to_csv('testers.csv', index=False)
    # Create playtesters df only
    df_playtesters = df_players[(df_players['class'] != '0')]
    # Drop class from df, since they are all players
    df_playtesters.drop(columns='class', axis=1, inplace=True)
    # Merge testers with players by email
    df_players_merged = pd.merge(df_testers, df_players, on='email')
    # Merge ingame player data with list of all players
    df_players_merged = pd.merge(df_players_merged,df_playtesters, on='player_id' )
    # Export gameplay df for model1
    df_players_merged.to_csv('gameplay_model.csv', index=False)
    return df_players_merged

#Players session stats df
def sessions():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ={api}
    bigquery_client = bigquery.Client(project={project})
    QUERY = """
    SELECT * FROM {session stats}
    WHERE ds > '2019-06-11'
    AND ds < '2020-03-01'
    """
    query_job = bigquery_client.query(QUERY)
    df_sessions = query_job.to_dataframe()
    df_sessions.to_csv('dataset_sessions.csv', index=False)
    return df_sessions

def sessions_dau(df_sessions):
    dau = df_sessions[['ds', 'total_daily_sessions_count']]
    dau = dau.rename(columns={'total_daily_sessions_count': 'DAU', 'ds': 'day'})
    plt.figure(figsize=(40,8))
    x = sns.lineplot(y="DAU", x='day', data=dau)
    x.set(yticklabels=[])
    x.set(xlabel='Date', ylabel='DAU')
    return plt.savefig('DAU', dpi=600)

def sessions_mau(df_sessions):
    mau = df_sessions[['total_daily_sessions_count', 'month']].groupby('month').agg('sum').reset_index()
    plt.figure(figsize=(40,8))
    x = sns.lineplot(y="MAU", x='day', data=mau)
    x.set(yticklabels=[])
    x.set(xlabel='Date', ylabel='MAU')
    return plt.savefig('MAU', dpi=600)

def sessions_stickiness(df_sessions):
    # Avg DAU
    stickiness_avg_dau = df_sessions[['avg_daily_sessions_count', 'month']].groupby('month').agg('mean').reset_index()
    # Total MAU
    stickiness_total_mau = df_sessions[['avg_daily_sessions_count', 'month']].groupby('month').agg('sum').reset_index()
    stickiness_total_mau.rename(columns={'avg_daily_sessions_count': 'total_mau'}, inplace=True)
    #Merge avg DAU and MAU by month
    stickiness = pd.merge(stickiness_total_mau,stickiness_avg_dau, on='month' )
    # Make datetime variables
    stickiness_avg_dau.day = pd.to_datetime(stickiness_avg_dau.day)
    # Change to months name, not number
    stickiness_avg_dau['month'] = stickiness_avg_dau.day.dt.month_name()
    # Merge
    stickiness = pd.merge(stickiness, stickiness_avg_dau[['DAU', 'month']].groupby('month').agg('mean').reset_index(), on='month')
    # Clean cols
    stickiness.drop(columns='avg_daily_sessions_count', axis=1, inplace=True)
    stickiness.rename(columns={'DAU': 'avg_dau'}, inplace=True)
    # Create col stickiness rate
    stickiness['stickiness_rate'] = (stickiness.avg_dau / stickiness.total_mau)
    stickiness.to_csv('stickiness.csv', index=False)
    # Plot
    plt.figure(figsize=(20,8))
    x = sns.lineplot(y="stickiness_rate", x='month', data=stickiness)
    x.set(yticklabels=[])
    x.set(xlabel='Month', ylabel='Stickiness Rate')
    return plt.savefig('Stickiness Rate', dpi=600)

def first_sessions_playtime_corr(df_playtesters):
    # Correlation between first session length and avg playtime per player
    x = sns.regplot(x="First Session Playtime", y="Average Playtime",ci=None, data=df_playtesters.query('`Total Days Played` > 1').drop(columns='player_id', axis=1))
    x.set(xticklabels=[])
    x.set(yticklabels=[])
    plt.savefig('Average Playtime vs First Session Playtime', dpi=600)
    return plt.savefig('Average Playtime vs First Session Playtime', dpi=600)
    
def weekday_playtime(df_sessions):
    y = df_sessions[['avg_daily_sessions_duration', 'weekday']].groupby('weekday').agg('mean').reset_index()
    plt.figure(figsize=(10,8))
    x = sns.barplot(x='weekday', y='avg_daily_sessions_duration', data=y.sort_values('avg_daily_sessions_duration', ascending=False))
    x.set(yticklabels=[])
    x.set(xlabel='Day of the Week', ylabel='Average Session Duration')
    plt.savefig('Average Sessions Playtime Per Day of the Week', dpi=600)
    return plt.savefig('Average Sessions Playtime Per Day of the Week', dpi=600)
    
def action_freq_by_players(df_sessions):
    # Categorize days played
    def days_played_cat(df_sessions):
        if df_sessions.days_played < 2:
            return 'Played for One Day'
        elif (df_sessions.days_played > 2) and (df_sessions.days_played <= 6):
            return 'Played for Less than a Week'
        elif df_sessions.days_played > 6:
            return 'Played for More than a Week'
    # Apply to df
    df_sessions['days_played_cat']=df_sessions.apply(lambda x: days_played_cat(x), axis=1)
    # Greater Avg playtime = 1, lower avg playtime = 0
    #Weapon usage in avg playtime category
    x = sns.catplot(x="avg_category", y="weapon", data=df_sessions,
                height=6, kind="bar", ci=None, aspect=1.5)
    plt.title('Weapon Usage By Players Classification', weight='bold', size=14)
    x.set(yticklabels=[])
    x.set(xlabel='', ylabel='Player Weapon Usage')
    plt.savefig('Weapon Usage By Players Classification', dpi=600)
    #Gathering usage in avg playtime category
    x = sns.catplot(x="avg_category", y="resources_gathered", data=df_sessions,
                height=6, kind="bar", ci=None, aspect=1.5)
    plt.title('Resource Gathering Frequency', weight='bold', size=14)
    x.set(yticklabels=[])
    x.set(xlabel='', ylabel='Player Resource Gathering Frequency')
    plt.savefig('Resource Gathering By Players Classification', dpi=600)
    # Building in avg playtime category
    x = sns.catplot(x="avg_category", y="outpost_buildings", data=df_sessions,
                height=6, kind="bar", ci=None, aspect=1.5)
    plt.title('Building Frequency By Players', weight='bold', size=14)
    x.set(yticklabels=[])
    x.set(xlabel='', ylabel='Building Frequency')
    plt.savefig('Building Frequency By Players Classification', dpi=600)
    
# Model Avg Playtime Prediction
    
df = pd.read_csv('gameplay_model.csv')    

def clean_df(df):
    # drop duplicates
    df = df.drop_duplicates()
    # Change class to wave
    df.rename(columns={'class': 'wave' }, inplace=True)
    # All ID are players with playtime, therefore wave are all players and useless ---> drop
    df.drop(columns='Wave', axis=1, inplace=True)
    # Drop uneeded cols
    df.drop(columns=['email','wave', 'avg_playtime'],inplace=True)
    #Relabel cols
    champion_dict = {'0.0':'No', '1.0':'Yes', 'Other': 'Unknown'}
    df.champion = df.champion.map(champion_dict)
    wow_dict = {0:'No', 1:'Yes'}
    df.WoW = df.WoW.map(wow_dict)
    ark_dict = {0:'No', 1:'Yes'}
    df.Ark = df.Ark.map(ark_dict)
    fortnite_dict = {0:'No', 1:'Yes'}
    df.Fortnite = df.Fortnite.map(fortnite_dict)
    return df
    
def scale_df(df):
    # Creat Lists of cat cols and list of num cols
    cat_cols = ['champion', 'us_canada', 'age', 'gender', 'platform', 'genre', 'pc',
           'community', 'WoW', 'Ark', 'Fortnite', 'pvp']
    num_columns = [i for i in df.drop(columns=['player_id', 'avg_category'],axis=1).columns if i not in cat_cols]
    # Scale all num cols
    scaler = StandardScaler()
    for i in num_columns:
        df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))
    return df, cat_cols, num_columns
    
def make_dummies(df, cat_cols):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
    
def logistic_regression_model(df):
    # Train test split/ 1/3 test size
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['player_id', 'avg_category'],
                                                             axis=1),
                                                   df.drop(columns=['player_id'],axis=1).avg_category, 
                                                    test_size=1/3)
    # Log regression Model
    model_LR = LogisticRegression(class_weight='balanced')
    res_LR = model_LR.fit(X_train,y_train)
    pred_LR = model_LR.predict(X_test)
    conf_LR = confusion_matrix(y_test,pred_LR)
    # Metrics
    print("Accuracy Score : ", accuracy_score(y_test,pred_LR))
    print("Recall Score : ", recall_score(y_test,pred_LR))
    print("Precision Score : ", precision_score(y_test,pred_LR))
    print("F1 Score : ", f1_score(y_test,pred_LR))
    print("Log Loss Score : ", log_loss(y_test, pred_LR))
    # ROC Graph
    model_LR_roc = roc_auc_score(y_test,pred_LR)
    fpr,tpr,thresholds = roc_curve(y_test, model_LR.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr,tpr,label=f'Model_LR (area={model_LR_roc})')
    plt.plot([0,1], [0,1])
    plt.title('Logistic Regression Model')
    plt.savefig('Logistic Regression Model', dpi=600)
    return df


# creation of a fucntion to run any model and display the results
def prediction_model_params(algorithm,training_x,testing_x,training_y,testing_y):
    # model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))
    print("Precision Score : ",precision_score(testing_y,predictions))
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    print ("Area under curve : ",model_roc_auc,"\n")
    fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])
    #plot confusion matrix
    plt.figure()
    matrix =sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    matrix.set_xticklabels(['Below Avg', 'Above Avg'])
    matrix.set_yticklabels(['Below Avg', 'Above Avg'])
    matrix.set_title('Confusion matrix')
    #plot roc curve
    fpr,tpr,thresholds=roc_curve(testing_y, probabilities[:,1])
    plt.figure()
    plt.plot(fpr,tpr,label=f'Roc (area={model_roc_auc})')
    plt.plot([0,1],[0,1])
    plt.legend()
    plt.title('Model performance')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()

# prediction_model_params(model_LR,X_train,X_test,y_train,y_test)

def compare_models(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['player_id', 'avg_category'],axis=1),df.drop(columns=['player_id'],axis=1).avg_category, test_size=1/3)
    # Models to test
    model_list=[KNeighborsClassifier(), GaussianNB(), 
            DecisionTreeClassifier(), 
            RandomForestClassifier(), 
            CatBoostClassifier()]
    # Loop through models, fit, train, and test model
    l_acc = []
    l_cm = []
    for model in model_list:
        model2=model.fit(X=X_train, y=y_train)
        y_pred2 = model2.predict(X_test)
        l_acc.append(accuracy_score(y_test,y_pred2))
        l_cm.append(confusion_matrix(y_test,y_pred2))
        print(type(model2).__name__, ' is done')
    # Return dataframe of models and accuracy metric to compare
    compare_models_df = pd.DataFrame([[type(i).__name__ for i in model_list],l_acc]).T.sort_values(by=1)
    return compare_models_df
    
def cat_boost_model(df, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['player_id', 'avg_category'],axis=1),df.drop(columns=['player_id'],axis=1).avg_category, test_size=1/3)
    # No dummies
    X = df.drop(columns=['avg_category'],axis=1)
    y = df.avg_category
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
    # Initialize data for catboost
    dummies = cat_cols
    cat_features = dummies
    # Initialize CatBoostClassifier & optimize precision
    model = CatBoostClassifier(iterations=1000,
                              eval_metric='Precision')
    model.fit(X_train, y_train, cat_features)
    # Get predicted classes
    preds_class = model.predict(X_test)
    # Metrics
    print("Accuracy Score : ", accuracy_score(y_test,preds_class))
    print("Recall Score : ", recall_score(y_test,preds_class))
    print("Precision Score : ", precision_score(y_test,preds_class))
    print("F1 Score : ", f1_score(y_test,preds_class))
    print("Log Loss Score : ", log_loss(y_test, preds_class))
    # ROC Graph
    conf_CB = confusion_matrix(y_test, preds_class)
    print(conf_CB)
    model_CB_roc = roc_auc_score(y_test,preds_class)
    print(model_CB_roc)
    # check new auc score and compare it to the first model
    fpr,tpr,_ = roc_curve(y_test, preds_class)
    # auc=roc_auc_score(y_test,preds_class)
    plt.grid(False)
    plt.plot(fpr,tpr, color= '#42ffb7',)
    plt.title('Cat Boost Prediction Model')
    plt.savefig('Cat Boost Prediction Model', dpi=600)
    return 

# Correlations

def gameplay_demographic_correlations():
    df = pd.read_csv('gameplay_model.csv')
    # drop duplicates
    df = df.drop_duplicates()
    # All ID are players with playtime, therefore wave are all players and useless ---> drop
    df.drop(columns='Wave', axis=1, inplace=True)
    # Change class to wave
    df.rename(columns={'class': 'wave' }, inplace=True)
    # Avg playtime by age
    plt.figure(figsize=(8,8))
    x = sns.barplot(x='age', y='avg_playtime', data=df, ci=False)
    x.set(yticklabels=[])
    plt.title('Average Session Playtime by Age', weight='bold', size=14)
    x.set(xlabel='Age', ylabel='Average Session Playtime')
    plt.savefig('Average Playtime by Age', dpi=600)
    
    #Avg session playtime by platform
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='platform', y='avg_playtime', data=df, ci=False)
    x.set(yticklabels=[])
    plt.title('Average Session Playtime by Platform Preference', weight='bold', size=14)
    x.set(xlabel='Preferred Platform', ylabel='Average Session Playtime')
    plt.savefig('Average Session Playtime by Platform Preference', dpi=600)
    
    #Avg sessions playtime by genre
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='genre', y='avg_playtime', data=df, ci=False)
    x.set(yticklabels=[])
    plt.title('Average Session Playtime by Genre Preference', weight='bold', size=14)
    x.set(xlabel='Preferred Genre', ylabel='Average Session Playtime')
    plt.savefig('Average Session Playtime by Genre Preference', dpi=600)
    
    # Avg playsession by community
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='community', y='avg_playtime', data=df, ci=False)
    x.set(yticklabels=[])
    #x.set(xticklabels=['Preferres PC', 'Does Not Prefer PC'])
    plt.title('Average Session Playtime by Community Activity', weight='bold', size=14)
    x.set(xlabel='Community Activity', ylabel='Average Session Playtime')
    plt.savefig('Average Session Playtime by Community Activity', dpi=600)
    
    # Avg session playtime by PvP pref
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='pvp', y='avg_playtime', data=df, ci=False)
    x.set(yticklabels=[])
    x.set(xticklabels=['Love', 'Indifferent', 'Dislike'])
    plt.title('Average Session Playtime by PvP Preference', weight='bold', size=14)
    x.set(xlabel='PvP Preference', ylabel='Average Session Playtime')
    plt.savefig('Average Session Playtime by PvP Preference', dpi=600)
    
    #Weapon usage freq by genre pref
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='genre', y='weapon', data=df, ci=False)
    x.set(yticklabels=[])
    plt.title('Weapon Usage Frequency by Genre Preference', weight='bold', size=14)
    x.set(xlabel='Players Genre Preference', ylabel='Weapon Usage Frequency')
    plt.savefig('Weapon Usage Frequency by Genre Preference', dpi=600)
    
    # Resource gathering freq by genre pref
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='genre', y='resources_gathered', data=df, ci=False)
    x.set(yticklabels=[])
    plt.title('Resource Gathering Frequency by Genre Preference', weight='bold', size=14)
    x.set(xlabel='Genre Preference', ylabel='Resource Gathering Frequency')
    plt.savefig('Resource Gathering Frequency by Genre Preference', dpi=600)
    
    # Bulilding Freq by genre pref
    plt.figure(figsize=(10,5))
    x = sns.barplot(x='genre', y='outpost_buildings', data=df, ci=False)
    x.set(yticklabels=[])
    plt.title('Building Frequency by Genre Preference', weight='bold', size=14)
    x.set(xlabel='Genre Preference', ylabel='Building Frequency')
    plt.savefig('Building Frequency by Genre Preference', dpi=600)
    return df

# Clustering Model

df = pd.read_csv('gameplay_model.csv')

def clean_cluster_model(df):
    # Drop cols
    df = df.drop(columns=['player_id', 'email','wave', 'avg_category' ])
    #clean cols
    champion_dict = {'0.0':'No', '1.0':'Yes', 'Other': 'Unknown'}
    df.champion = df.champion.map(champion_dict)
    wow_dict = {0:'No', 1:'Yes'}
    df.WoW = df.WoW.map(wow_dict)
    ark_dict = {0:'No', 1:'Yes'}
    df.Ark = df.Ark.map(ark_dict)
    fortnite_dict = {0:'No', 1:'Yes'}
    df.Fortnite = df.Fortnite.map(fortnite_dict)
    return df

def normalize_cluster_model(df):
    # Creat Lists of cat cols and list of num cols
    cat_cols = ['champion', 'us_canada', 'age', 'gender', 'platform', 'genre', 'pc',
           'community', 'WoW', 'Ark', 'Fortnite', 'pvp']
    num_columns = [i for i in df.columns if i not in cat_cols]
    # Scale all num cols
    scaler = StandardScaler()
    for i in num_columns:
        df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))
    # Create dummies
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

# KMeans Model

# KEblow Optiization
def kelbow_optimization(df):
    # Shows optimal number of clusters for model
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,10))
    visualizer.fit(df)
    visualizer.poof()
    visualizer.show(outpath="Elbow Kmeans Cluster.pdf")
    return df

def kmeans_cluster(df):
    X = df
    model = KMeans(n_clusters=3)
    model.fit(X)
    y_pred = model.predict(X)
    print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
    print('Silhouette Score ', silhouette_score(X,y_pred))
    # Visualize clusters
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c=['#42ffb7', '#00ffff', '#828cfb',], s=200, alpha=0.5)
    plt.title('K Means Cluster Epicenters')
    plt.savefig('K Means Cluster Epicenter', dpi=600)
    return df

def agglo_optimization(df):
    X = df
    # Find best linkage
    metric_list = ['complete','single', 'average', 'ward']
    for i in metric_list:
        single = AgglomerativeClustering(n_clusters=4, linkage=i)
        y_pred = single.fit_predict(X)
        print('Metric ', i)
        print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
        print('Silhouette Score ', silhouette_score(X,y_pred))
    # Find best num of clusters
    for i in range(2,10):
        single = AgglomerativeClustering(n_clusters=i, linkage='average')
        y_pred = single.fit_predict(X)
        print('Metric ', i)
        print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
        print('Silhouette Score ', silhouette_score(X,y_pred))
    return df

def agglomerative_cluster(df):
    # Agglomerative
    model = AgglomerativeClustering(n_clusters=4, linkage='average', affinity='euclidean')
    model.fit(df)
    y_pred = model.fit_predict(df)
    print(davies_bouldin_score(df,y_pred))
    print(silhouette_score(df,y_pred))
    # Add clustering to df
    df['Cluster'] = model.labels_
    return df

def vis_clustering_model(df):
    # Avg playtime by age group
    age_avg_playtime = pd.pivot_table(data=df,index=('Cluster', 'age'),values='avg_playtime', aggfunc=('mean'))
    age_avg_playtime = age_avg_playtime.reset_index()
    ax = sns.catplot(x="Cluster", y="avg_playtime", kind="bar", hue='age', data=age_avg_playtime, aspect=1.5)
    plt.title('Average Playtime by Age Group', weight='bold', size=14)
    ax.set(ylabel='Avgerage Playtime in Minuets')
    ax.set(xlabel='Player Clusters')
    plt.savefig('Average Playtime by Age Group and Cluster', dpi=600)
    
    # Avg playtime by genre
    genre_avg_playtime = pd.pivot_table(data=df,index=('Cluster', 'genre'),values='avg_playtime', aggfunc=('mean'))
    genre_avg_playtime = genre_avg_playtime.reset_index()
    ax = sns.catplot(x="Cluster", y="avg_playtime", kind="bar", hue='genre', data=genre_avg_playtime, aspect=1.5)
    plt.title('Average Playtime by Genre Group', weight='bold', size=14)
    ax.set(ylabel='Avgerage Playtime in Minuets')
    ax.set(xlabel='Player Clusters')
    plt.legend(title="Preferred Genre", fancybox=True, loc =2)
    plt.savefig('Average Playtime by Players Preferred Genre and Cluster', dpi=600)
    
    # Avg playtime by Pvp pref
    pvp_avg_playtime = pd.pivot_table(data=df,index=('Cluster', 'pvp'),values='avg_playtime', aggfunc=('mean'))
    pvp_avg_playtime = pvp_avg_playtime.reset_index()
    ax = sns.catplot(x="Cluster", y="avg_playtime", kind="bar", hue='pvp', data=pvp_avg_playtime, aspect=1.5)
    plt.title('Average Playtime by PvP Preference', weight='bold', size=14)
    ax.set(ylabel='Avgerage Playtime in Minuets')
    ax.set(xlabel='Player Clusters')
    plt.savefig('Average Playtime by PvP Preference and Cluster', dpi=600)
    
    # Resources gathered by community
    community_res = pd.pivot_table(data=df,index=('Cluster', 'community'),values='resources_gathered', aggfunc=('mean'))
    community_res = community_res.reset_index()
    ax = sns.catplot(x="Cluster", y="resources_gathered", kind="bar", hue='community', data=community_res, aspect=1.5)
    plt.title('Resources Gathered by Community Activity', weight='bold', size=14)
    ax.set(ylabel='Resources Gathered in Event Counts')
    ax.set(xlabel='Player Clusters')
    plt.savefig('Resources Gathered by Community Activity', dpi=600)
    return df