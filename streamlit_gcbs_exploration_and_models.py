############## Imports ##############

#importing libraries
# web-app
import streamlit as st
# data manipulation
import itertools
import pandas as pd
import numpy as np
import json
import requests
# stat
import pingouin as pg
import researchpy as rp
from scipy.stats import shapiro, jarque_bera, normaltest, skew, kurtosis, spearmanr
# viz
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
# ml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load

#setting options for better visualization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)

############## Loading the dataset ##############

#loading the dataset
gcbs_df = pd.read_csv('https://raw.githubusercontent.com/bolps/conspiracy/main/openpsychometrics_gcbs_data.csv')

############## Data munging ##############

# dropping 'introelapse','testelapse','surveyelapse' (overall time spent in every block) columns as I'm not going to use them
gcbs_df = gcbs_df.drop(['introelapse','testelapse','surveyelapse'], axis=1)

# download/parsing json labels
survey_labels_json = requests.get("https://raw.githubusercontent.com/bolps/conspiracy/main/survey_labels.json")
survey_labels_dict = json.loads(survey_labels_json.text)
# mapping labels on dataframe, converting 'Missing' label to NaN, casting to 'category' (or ordered category)
for column in ['education','urban','gender','engnat','hand','religion','orientation','race','voted','married']:
    gcbs_df[column] = gcbs_df[column].astype(str).map(survey_labels_dict[column])
    gcbs_df[column] = gcbs_df[column].replace('Missing', np.nan)
    gcbs_df[column] = gcbs_df[column].astype("category")
gcbs_df['education'] = gcbs_df['education'].cat.reorder_categories(['Less than high school', 'High school', 'University degree', 'Graduate degree'], ordered=True)

# assigning NaN to values which makes no sense (i.e. too old or too young people)
gcbs_df['familysize'] = gcbs_df['familysize'].replace(0, np.nan)
gcbs_df.loc[gcbs_df.familysize > 10, 'familysize'] = np.nan
gcbs_df.loc[gcbs_df.age > 90, 'age'] = np.nan
gcbs_df.loc[gcbs_df.age < 10, 'age'] = np.nan
gcbs_df.loc[gcbs_df.education < 'University degree', 'major'] = np.nan

# recoding family size in groups
gcbs_df['familytype'] = gcbs_df['familysize']
gcbs_df['familytype'] = pd.cut(gcbs_df.familytype,bins=[0,1,3,5,10],labels=['Small','Medium','Large','Very Large'])
# recoding age in groups
gcbs_df['agegroup'] = gcbs_df['age']
gcbs_df['agegroup'] = pd.cut(gcbs_df.agegroup,bins=[0,17,25,45,65,90],labels=['Adolescence','Emerging Adulthood','Early Adulthood','Middle Adulthood','Late Adulthood'])

# grouping christians in religion (to avoid excessive fragmentation)
christian_groups = [x for x in list(gcbs_df.religion.value_counts().index) if 'Christian' in x]
for label in christian_groups:
    gcbs_df['religion'] = gcbs_df['religion'].replace(label, 'Christian')
# merging demographic groups with few people (<=100) with 'Other'
for column in ['religion','orientation','race']:
    minorities = list(gcbs_df[column].value_counts()[gcbs_df[column].value_counts()<= 100].index)
    for label in minorities:
        gcbs_df[column] = gcbs_df[column].replace(label, 'Other')

# Normalizing text for the field major (free-form). Example: '           aCTING' = 'Acting', 'ACTING' = 'Acting'
gcbs_df['major'] = gcbs_df['major'].map(lambda x : x.strip().capitalize() if isinstance(x, str) else np.nan)

# download/parsing json labels. Major labels has been manually grouped in macro-categories
major_labels_json = requests.get("https://raw.githubusercontent.com/bolps/conspiracy/main/major_labels.json")
major_labels_dict = json.loads(major_labels_json.text)
# mapping labels on dataframe in order to crate a new column with the 'major cluster'. Example: Graphic design (Major) - Arts (Cluster) 
gcbs_df['major_cluster'] = gcbs_df['major'].map(major_labels_dict['Cluster'])
gcbs_df['major_cluster'] = gcbs_df['major_cluster'].replace('Missing', np.nan)
gcbs_df['major_cluster'] = gcbs_df['major_cluster'].astype("category")

# renaming GCBS items for consistency and readability (columns Q1-Q15 belongs to GCBS scale). Example: Q1 becomes GCBS1
gcbs_df.columns = [column.replace('Q', 'GCBS') for column in gcbs_df.columns]

# assigning NaN to GCBS and TIPI missing values
tipi_gcbs_cols = list(gcbs_df.filter(like='TIPI',axis=1).columns) + list(gcbs_df.filter(like='GCBS',axis=1).columns)
for column in tipi_gcbs_cols:
    gcbs_df[column] = gcbs_df[column].replace(0, np.nan)
# dropping responses which contain missing data from GCBS and TIPI
gcbs_df = gcbs_df.dropna(subset=tipi_gcbs_cols)

# computing scores for GCBS scale according to the codebook. Note: GCBS scale doesn't use reverse items.
gcbs_cols = list(gcbs_df.filter(like='GCBS',axis=1).columns)
gcbs_df['GCBS_Overall'] = (gcbs_df[gcbs_cols].sum(axis=1))/len(gcbs_cols)
# computing scores for TIPI (personality) scale and subscales according to the codebook. Note: Some of the items are reversed (formula for reversing: reverse = (number_of_levels + 1) - raw_value).
reverse = lambda x : (8-x)
gcbs_df['TIPI_Extraversion'] = (gcbs_df['TIPI1'] + reverse(gcbs_df['TIPI6']))/2
gcbs_df['TIPI_Agreeableness'] = (reverse(gcbs_df['TIPI2']) + gcbs_df['TIPI7'])/2
gcbs_df['TIPI_Conscientiousness'] = (gcbs_df['TIPI3'] + reverse(gcbs_df['TIPI8']))/2
gcbs_df['TIPI_Emotional_Stability'] = (reverse(gcbs_df['TIPI4']) + gcbs_df['TIPI9'])/2
gcbs_df['TIPI_Openness'] = (gcbs_df['TIPI5'] + reverse(gcbs_df['TIPI10']))/2

# computing variable for counting validity check errors (validity check items are simple questions used to probe user attention and engagement)
gcbs_df['SURV_ValidityCheck_Errors'] = gcbs_df['VCL6'] + gcbs_df['VCL9'] + gcbs_df['VCL12']
gcbs_df = gcbs_df.drop(['VCL1','VCL2','VCL3','VCL4','VCL5','VCL6','VCL7','VCL8','VCL9','VCL10','VCL11','VCL12','VCL13','VCL14','VCL15','VCL16'], axis=1)
# computing variable to spot straightliners for GCBS (respondents who selected the same answer to all items of the scale).
# the respondent is a straightliner if the std deviation of the scale is 0 (always the same answer)
gcbs_df['SURV_CheckStraightliners_GCBS'] = gcbs_df[['GCBS1','GCBS2','GCBS3','GCBS4','GCBS5','GCBS6','GCBS7','GCBS8','GCBS9','GCBS10','GCBS11','GCBS12','GCBS13','GCBS14','GCBS15']].std(axis=1)
gcbs_df['SURV_CheckStraightliners_GCBS'] = gcbs_df['SURV_CheckStraightliners_GCBS'] == 0
# computing variable to spot straightliners for TIPI (respondents who selected the same answer to all items of the scale).
# the respondent is a straightliner if the std deviation of the scale is 0 (always the same answer)
gcbs_df['SURV_CheckStraightliners_TIPI'] = gcbs_df[['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10']].std(axis=1)
gcbs_df['SURV_CheckStraightliners_TIPI'] = gcbs_df['SURV_CheckStraightliners_TIPI'] == 0
# computing variable to spot speed responses for GCBS (the core construct measured). Note: Unfortunately we have no data on TIPI response times.
# I considered a speed response if the response time of at least one item of the scale is less than the 5th quantile. Note: I dropped response times as I don't need them anymore
for item in ['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15']:
    gcbs_df[item] = gcbs_df[item] < gcbs_df[item].quantile(.05)
gcbs_df['SURV_QuickResponse_GCBS'] = gcbs_df[['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15']].any(axis=1)
gcbs_df = gcbs_df.drop(['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15'], axis=1)
# renaming columns for better organization.
# I logically divided data in blocks: RAW (raw GCBS and TIPI data), DEMO (demographics), GCBS (gcbs scores), TIPI (tipi scores) and SURV (survey related info)
gcbs_df.columns = [('RAW_'+column) for column in gcbs_df.columns[:gcbs_df.columns.get_loc('TIPI10')+1]] + [('DEMO_'+column) for column in gcbs_df.columns[gcbs_df.columns.get_loc('TIPI10')+1:gcbs_df.columns.get_loc('major_cluster')+1]] + list(gcbs_df.columns[gcbs_df.columns.get_loc('major_cluster')+1:])

############## Data cleaning ##############

# defining masks for removing non-valid responses, straightliners and quick responders
validity_check_erros_mask = gcbs_df['SURV_ValidityCheck_Errors'] == 0
check_straightliners_gcbs_mask = gcbs_df['SURV_CheckStraightliners_GCBS'] == False
check_straightliners_tipi_mask = gcbs_df['SURV_CheckStraightliners_TIPI'] == False
quick_response_gcbs_mask = gcbs_df['SURV_QuickResponse_GCBS'] == False

# applying masks and dropping SURV_* columns used for cleaning and no longer needed
gcbs_clean_df = gcbs_df[validity_check_erros_mask & check_straightliners_gcbs_mask & check_straightliners_tipi_mask & quick_response_gcbs_mask]
gcbs_clean_df = gcbs_clean_df.reset_index(drop=True)
gcbs_clean_df = gcbs_clean_df.drop(['SURV_ValidityCheck_Errors','SURV_CheckStraightliners_GCBS','SURV_CheckStraightliners_TIPI','SURV_QuickResponse_GCBS'], axis=1)

############## Data quality ##############

def getCronbachDict(scale_name,df, ci=.95):
    cron = pg.cronbach_alpha(data=df)
    
    if cron[0] >= 0.9:
        internal_consistency = 'Excellent'
    elif cron[0] >= 0.8:
        internal_consistency = 'Good'
    elif cron[0] >= 0.7:
        internal_consistency = 'Acceptable'
    elif cron[0] >= 0.6:
        internal_consistency = 'Questionable'
    elif cron[0] >= 0.5:
        internal_consistency = 'Poor'
    else:
        internal_consistency = 'Unacceptable'

    if len(df.columns) == 2:
        note = 'When items = 2, coefficient alpha almost always underestimates true reliability'
    else:
        note = ''
    
    return {
        'scale':scale_name,
        'scale_items':len(df.columns),
        'sample_size':len(df),
        'cronbach_alpha':round(cron[0],3),
        'ci_lower':cron[1][0],
        'ci_upper':cron[1][1],
        'ci':ci,
        'internal_consistency':internal_consistency,
        'note':note
  }

cronbach_dict_list = []
# computing cronbach alpha for GCBS scale
cronbach_dict_list.append(getCronbachDict('GCBS_Overall',gcbs_clean_df[['RAW_GCBS1','RAW_GCBS2','RAW_GCBS3','RAW_GCBS4','RAW_GCBS5','RAW_GCBS6','RAW_GCBS7','RAW_GCBS8','RAW_GCBS9','RAW_GCBS10','RAW_GCBS11','RAW_GCBS12','RAW_GCBS13','RAW_GCBS14','RAW_GCBS15']]))
# in order to compute cronbach alpha for TIPI scale and subscales i need columns for reverse items (it's necessary to reverse the scores before computing ca)
for column in ['RAW_TIPI2','RAW_TIPI4','RAW_TIPI6','RAW_TIPI8','RAW_TIPI10']:
    gcbs_clean_df['{}_REV'.format(column)] = reverse(gcbs_clean_df[column])
# computing cronbach alpha for TIPI scale and subscales
cronbach_dict_list.append(getCronbachDict('TIPI_Overall',gcbs_clean_df[['RAW_TIPI1','RAW_TIPI2_REV','RAW_TIPI3','RAW_TIPI4_REV','RAW_TIPI5','RAW_TIPI6_REV','RAW_TIPI7','RAW_TIPI8_REV','RAW_TIPI9','RAW_TIPI10_REV']]))
cronbach_dict_list.append(getCronbachDict('TIPI_Extraversion',gcbs_clean_df[['RAW_TIPI1','RAW_TIPI6_REV']]))
cronbach_dict_list.append(getCronbachDict('TIPI_Agreeableness',gcbs_clean_df[['RAW_TIPI2_REV','RAW_TIPI7']]))
cronbach_dict_list.append(getCronbachDict('TIPI_Conscientiousness',gcbs_clean_df[['RAW_TIPI3','RAW_TIPI8_REV']]))
cronbach_dict_list.append(getCronbachDict('TIPI_Emotional_Stability',gcbs_clean_df[['RAW_TIPI4_REV','RAW_TIPI9']]))
cronbach_dict_list.append(getCronbachDict('TIPI_Openness',gcbs_clean_df[['RAW_TIPI5','RAW_TIPI10_REV']]))
# dropping all TIPI*_REV columns no longer needed (used for ca computation)
gcbs_clean_df = gcbs_clean_df.drop(list(gcbs_clean_df.filter(like='_REV',axis=1).columns), axis=1)
# building a table with cronbach alpha info
cronbach_df  = pd.DataFrame(cronbach_dict_list)

############## Data Exploration ##############

# function for pie charts
def pieChart(df, col, title='', subtitle=''):
    total_responses = sum(df[col].value_counts()[df[col].value_counts()> 0])
    count_df = df[col].value_counts()[df[col].value_counts()> 0].rename_axis(col).reset_index(name='counts')
    fig = count_df.iplot(kind='pie', labels=col, values='counts', hoverinfo="label+percent+name",hole=0.3, theme='white', asFigure=True)
    fig.update_traces(texttemplate='%{percent:.2%}')
    fig.update_layout(title_text='{} (N={})'.format(title,total_responses), title_x=0.1, legend=dict(orientation="h", xanchor = "center",  x = 0.5))
    return fig
# function for barcharts
def barChart(df, col, title='', subtitle=''):
    total_responses = sum(df[col].value_counts())
    fig = df[col].value_counts().sort_index(ascending=True).iplot(kind='bar', title='{} (N={})'.format(title,total_responses), color='rgb(195, 155, 211)', theme='white', asFigure=True)
    return fig

# plotting demographics
barChart(gcbs_clean_df, col='DEMO_agegroup', title='Age groups')
barChart(gcbs_clean_df, col='DEMO_education', title='Education')
barChart(gcbs_clean_df, col='DEMO_familytype', title='Family type')
pieChart(gcbs_clean_df, col='DEMO_urban', title='Area')
pieChart(gcbs_clean_df, col='DEMO_gender', title='Gender')
pieChart(gcbs_clean_df, col='DEMO_engnat', title='Language')
pieChart(gcbs_clean_df, col='DEMO_hand', title='Hand preference')
pieChart(gcbs_clean_df, col='DEMO_religion', title='Religion')
pieChart(gcbs_clean_df, col='DEMO_orientation', title='Sexual orientation')
pieChart(gcbs_clean_df, col='DEMO_race', title='Racial identification')
pieChart(gcbs_clean_df, col='DEMO_voted', title='Voted')
pieChart(gcbs_clean_df, col='DEMO_married', title='Married')
pieChart(gcbs_clean_df, col='DEMO_major_cluster', title='Major')

# decriptive statistics for scales
scales_list = ['GCBS_Overall','TIPI_Extraversion','TIPI_Agreeableness','TIPI_Conscientiousness','TIPI_Emotional_Stability','TIPI_Openness']
round(gcbs_clean_df[scales_list].describe(),2) ###!!!! remeber to print it in the web app
# plotting GCBS
gcbs_clean_df['GCBS_Overall'].iplot(kind='hist', opacity=0.75, color='rgb(12, 128, 128)', title='Generic Conspiracist Beliefs Scale Distribution', yTitle='Count', xTitle='GCBS score (overall)', bargap = 0, theme='white')
# Plotting Personality (TIPI) dimensions
gcbs_clean_df['TIPI_Extraversion'].iplot(kind='hist', opacity=0.75, color='rgb(93, 173, 226)', title='Personality - Extraversion Distribution', yTitle='Count', xTitle='TIPI score (extraversion)', bargap = 0, theme='white')
gcbs_clean_df['TIPI_Agreeableness'].iplot(kind='hist', opacity=0.75, color='rgb(72, 201, 176)', title='Personality - Agreeableness Distribution', yTitle='Count', xTitle='TIPI score (agreeableness)', bargap = 0, theme='white')
gcbs_clean_df['TIPI_Conscientiousness'].iplot(kind='hist', opacity=0.75, color='rgb(175, 122, 197)', title='Personality - Conscientiousness Distribution', yTitle='Count', xTitle='TIPI score (conscientiousness)', bargap = 0, theme='white')
gcbs_clean_df['TIPI_Emotional_Stability'].iplot(kind='hist', opacity=0.75, color='rgb(247, 220, 111)', title='Personality - Emotional Stability Distribution', yTitle='Count', xTitle='TIPI score (emotional stability)', bargap = 0, theme='white')
gcbs_clean_df['TIPI_Openness'].iplot(kind='hist', opacity=0.75, color='rgb(236, 112, 99)', title='Personality - Openness Distribution', yTitle='Count', xTitle='TIPI score (openness)', bargap = 0, theme='white')

# defining a function for normality tests 
def testNormality(x):
    #Shapiro-Wilk
    w, p_w = shapiro(x)
    #Jarque-Bera
    jb, p_jb = jarque_bera(x)
    #D’Agostino-Pearson
    k2, p_k2 = normaltest(x)
    
    #Additional info
    #Skewness
    s = skew(x)
    #Kurtosis
    k = kurtosis(x)
    
    return {
        'Shapiro (w)':w,
        'p-value (Shapiro)':p_w,
        'Jarque-Bera (jb)':jb,
        'p-value (Jarque-Bera)':p_jb,
        'D’Agostino-Pearson (k2)':k2,
        'p-value (D’Agostino-Pearson)':p_k2,
        'skewness':s,
        'kurtosis':k
    }

# testing normality for each scale and adding them to a pandas dataframe for displaying 
normality_check_list = []
for column in scales_list:
    distribution_info = testNormality(gcbs_clean_df[column])
    normality_check_list.append(distribution_info)
normality_tests_df = pd.DataFrame(normality_check_list)
normality_tests_df.index = scales_list
normality_tests_df ###!!!!! remeber to print it in the web app

############## Research questions  ##############

### Is the belief in conspiracy toeries influenced by certain socio-demographic groups?
demo_list = ['DEMO_agegroup','DEMO_familytype','DEMO_education','DEMO_urban','DEMO_gender','DEMO_engnat','DEMO_hand','DEMO_religion','DEMO_orientation','DEMO_race','DEMO_voted','DEMO_married','DEMO_major_cluster']

kruskallwallis_demo_gcbs_list = []
for demo in demo_list:
    result = pg.kruskal(data=gcbs_clean_df, dv='GCBS_Overall', between=demo, detailed=False)
    if result.loc['Kruskal']['p-unc']<.05:
        sign = 'Statistically significant result'
    else:
        sign = ''
    kruskallwallis_demo_gcbs_list.append((result.loc['Kruskal']['Source'],'GCBS_Overall',result.loc['Kruskal']['H'],result.loc['Kruskal']['p-unc'],result.loc['Kruskal']['ddof1'],sign))
kruskallwallis_df = pd.DataFrame(kruskallwallis_demo_gcbs_list, columns = ['Factor', 'Scale', 'H', 'p-value', 'DoF','Note'])
kruskallwallis_df ###!!!!! remeber to print it in the web app

sns.set_theme()
fig, axes = plt.subplots(4,2,figsize=(16,28))

axes[0,0].set_title('GCBS score by age group')
sns.swarmplot(ax=axes[0, 0], x="DEMO_agegroup", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[0, 0], x="DEMO_agegroup", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=45)
box.set_xlabel("Age group")
box.set_ylabel("GCBS score (overall)")

axes[0,1].set_title('GCBS score by education')
sns.swarmplot(ax=axes[0, 1], x="DEMO_education", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[0, 1], x="DEMO_education", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=45)
box.set_xlabel("Education")
box.set_ylabel("GCBS score (overall)")


axes[1,0].set_title('GCBS score by type of area')
sns.swarmplot(ax=axes[1, 0], x="DEMO_urban", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[1, 0], x="DEMO_urban", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=45)
box.set_xlabel("Type of area")
box.set_ylabel("GCBS score (overall)")

axes[1,1].set_title('GCBS score by gender')
sns.swarmplot(ax=axes[1, 1], x="DEMO_gender", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[1, 1], x="DEMO_gender", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=45)
box.set_xlabel("Gender")
box.set_ylabel("GCBS score (overall)")

axes[2,0].set_title('GCBS score by religion')
sns.swarmplot(ax=axes[2, 0], x="DEMO_religion", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[2, 0], x="DEMO_religion", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=90)
box.set_xlabel("Religion")
box.set_ylabel("GCBS score (overall)")

axes[2,1].set_title('GCBS score by racial identification')
sns.swarmplot(ax=axes[2, 1], x="DEMO_race", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[2, 1], x="DEMO_race", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=90)
box.set_xlabel("Racial identification")
box.set_ylabel("GCBS score (overall)")

axes[3,0].set_title('GCBS score by voted')
sns.swarmplot(ax=axes[3, 0], x="DEMO_voted", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[3, 0], x="DEMO_voted", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=0)
box.set_xlabel("Voted")
box.set_ylabel("GCBS score (overall)")

axes[3,1].set_title('GCBS score by college major cluster')
sns.swarmplot(ax=axes[3, 1], x="DEMO_major_cluster", y="GCBS_Overall", data=gcbs_clean_df, color=".2", size=1)
box = sns.boxplot(ax=axes[3, 1], x="DEMO_major_cluster", y="GCBS_Overall", data=gcbs_clean_df, palette="Set2",  width=0.4)
box.set_xticklabels(box.get_xticklabels(), rotation=90)
box.set_xlabel("College major (cluster)")
box.set_ylabel("GCBS score (overall)")

# set the spacing between subplots
fig.tight_layout()
plt.show()


demo_sign = ['DEMO_agegroup','DEMO_education','DEMO_urban','DEMO_gender','DEMO_religion','DEMO_race','DEMO_voted','DEMO_major_cluster']

chi_square_list = []
for pair in list(itertools.combinations(demo_sign,2)):
    crosstab, test_results, expected = rp.crosstab(gcbs_clean_df[pair[0]], gcbs_clean_df[pair[1]], test= "chi-square",expected_freqs= True,prop= "cell")
    
    chi_square = test_results.iloc[0]['results']
    p_value = test_results.iloc[1]['results']
    cramer_v = test_results.iloc[2]['results']
    
    if cramer_v > .25:
        assoc = 'Very strong'
    elif cramer_v > .15:
        assoc = 'Strong'
    elif cramer_v > .10:
        assoc = 'Moderate'
    elif cramer_v > .05:
        assoc = 'Weak'
    else:
        assoc = 'No or very weak'
        
    if p_value < .05:
        sign = 'Statistically significant result'
    else:
        sign = ''
        assoc = ''
    
    chi_square_list.append(('{} - {}'.format(pair[0],pair[1]),chi_square, p_value, cramer_v, sign, assoc))

chi_square_df = pd.DataFrame(chi_square_list, columns = ['Variables', 'χ2', 'p-value', "Cramer's V", 'Note','Degree of association'])
chi_square_df ###!!!!! remeber to print it in the web app

### Are there relationships between personality traits and belief in conspiracy theories?

correlation_list = []
for personality_dimension in list(gcbs_clean_df.filter(like='TIPI_',axis=1).columns):
        r_spear, p_spear = spearmanr(gcbs_clean_df[personality_dimension],gcbs_clean_df['GCBS_Overall'])
        
        if abs(r_spear) > .9:
            assoc = 'Very high correlation'
        elif abs(r_spear) > .7:
            assoc = 'High correlation'
        elif abs(r_spear) > .5:
            assoc = 'Moderate correlation'
        elif abs(r_spear) > .3:
            assoc = 'Low correlation'
        else:
            assoc = 'Negligible correlation'

        if p_spear < .05:
            sign = 'Statistically significant result'
        else:
            sign = ''
            assoc = ''
        
        correlation_list.append(('{} - GCBS_Overall'.format(personality_dimension), r_spear, p_spear, sign, assoc))

personality_conspiracy_corr_df = pd.DataFrame(correlation_list, columns=['variables', 'r_spearman', 'p_value','note','interpretation'])
personality_conspiracy_corr_df ###!!!!! remeber to print it in the web app

### Are there personality configurations that influence the belief in conspiracy theories?

# for model training see python notebook! (not included as training took hours)

# loading scaler/model
scaler = load('tipi_scaler.joblib') 
model = load('tipi_kmeans_model.joblib')

personality_df =  gcbs_clean_df[list(gcbs_clean_df.filter(like='TIPI_',axis=1).columns)]
personality_df
# applying the scaler to the dataset
scaled_array = scaler.transform(personality_df)
scaled_dataframe = pd.DataFrame(scaled_array, columns = personality_df.columns )
scaled_dataframe.head(5)
# cluster prediction
predicted_clusters = model.predict(scaled_dataframe)
gcbs_clean_df['TIPI_Personality_Cluster'] = predicted_clusters

gcbs_clean_df[['GCBS_Overall','TIPI_Extraversion','TIPI_Agreeableness','TIPI_Conscientiousness','TIPI_Emotional_Stability','TIPI_Openness','TIPI_Personality_Cluster']].head(5) #preview ###!!!!! remeber to print it in the web app
pg.kruskal(data=gcbs_clean_df, dv='GCBS_Overall', between='TIPI_Personality_Cluster', detailed=False) ###!!!!! remeber to print it in the web app

