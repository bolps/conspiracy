############## Imports ##############

#importing libraries
# web-app
import io
import streamlit as st
# data manipulation
import itertools
import random
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

############## Web App Intro ##############

st.sidebar.markdown("[Background](#background)", unsafe_allow_html=True)
st.sidebar.markdown("[Dataset](#dataset)", unsafe_allow_html=True)
st.sidebar.markdown("[Data preprocessing](#data-preprocessing)", unsafe_allow_html=True)
st.sidebar.markdown("[Data cleaning](#data-cleaning)", unsafe_allow_html=True)
st.sidebar.markdown("[Data quality](#data-quality)", unsafe_allow_html=True)
st.sidebar.markdown("[Partecipants](#partecipants)", unsafe_allow_html=True)
st.sidebar.markdown("[Scales](#scales)", unsafe_allow_html=True)
st.sidebar.markdown("[Research question](#research-question)", unsafe_allow_html=True)
st.sidebar.markdown("[Conspiracy theories and demographics](#conspiracy-theories-and-demographics)", unsafe_allow_html=True)
st.sidebar.markdown("[Conspiracy theories and personality traits](#conspiracy-theories-and-personality-traits)", unsafe_allow_html=True)
st.sidebar.markdown("[Conspiracy theories and personality configurations](#conspiracy-theories-and-personality-configurations)", unsafe_allow_html=True)


st.header('Who are the Conspiracy Theorists?')
st.image('https://media.newyorker.com/photos/5cae57045d9cdc3d91d43009/2:1/w_2559,h_1279,c_limit/190422_r34133.jpg', caption ='Illustration by Zohar Lazar - The New Yorker')
st.subheader('Background')
st.write('Today, more than ever, conspiracy theories play a central role in the social research community. Even though conspiracy theories have many facets, a conspiracist belief is “the unnecessary assumption of conspiracy when other explanations are more probable” (Aaronovitch, 2010, p. 5). Throughout history, conspiracy theories had many negative outcomes such as the anti-vax movement or the post-truth communication endorsed by some politicians. However, few studies investigated in deeply the characteristics of people who embrace such views.')
st.write('Demographic research found relationships between income, education, religion (Smallpage et al., 2020) and beliefs in conspiracy theories, while psychological studies have highlighted relationships between conspiratorial thinking and psychopathological traits such as paranoia and Machiavellianism (Brotherton et al., 2013; Douglas & Sutton, 2011). On the other hand, when it comes to normal personality traits, relationships are still largely unknown. Just a few studies reported weak but significant relationships between belief in conspiracy theories and two personality traits: openness and agreeableness but other studies have failed to replicate the results (Brotherton et al., 2013).')
st.write('This research aims to explore and further investigate the relationships between belief in conspiracy theories, demographics, and personality.')
if st.checkbox('Show reference'):
    st.write('>Aaronovitch, D. (2010). *Voodoo Histories: The Role of the Conspiracy Theory in Shaping Modern History.* London: Jonathan Cape.')
    st.write('>Brotherton, R., French, C. C., & Pickering, A. D. (2013). *Measuring belief in conspiracy theories: The generic conspiracist beliefs scale.* Frontiers in psychology, 4, 279.')
    st.write('>Douglas, K. M., & Sutton, R. M. (2011). *Does it take one to know one? Endorsement of conspiracy theories is influenced by personal willingness to conspire.* British Journal of Social Psychology, 50(3), 544-552.')
    st.write('>Smallpage, S. M., Drochon, H., Uscinski, J. E., & Klofstad, C. (2020). *Who are the Conspiracy Theorists?: Demographics and conspiracy theories.* In Routledge handbook of conspiracy theories (pp. 263-277). Routledge.')

st.subheader('Dataset')
st.image('https://i.ibb.co/HPfjnwd/8.png')

st.write('This data was collected in 2016 through an interactive on-line version of the *Generic Conspiracist Beliefs Scale* (Brotherton et al. 2013) along with the *Ten Item Personality Inventory* (Gosling et al. 2003) and demographics. Visitors completed the test primarily for personal amusement. At the end of the test but before the results were displayed, users were asked if they would be willing to complete an additional survey and allow their responses to be saved for research. Finally data was published on [Kaggle]("https://www.kaggle.com/yamqwe/measuring-belief-in-conspiracy-theories") as a part of the *Open Psychometrics Dataset Collection*.')
st.write('>Brotherton, R., French, C. C., & Pickering, A. D. (2013). Measuring belief in conspiracy theories: The generic conspiracist beliefs scale. *Frontiers in psychology*, 4, 279.')
st.write('>Gosling, S. D., Rentfrow, P. J., & Swann Jr, W. B. (2003). A very brief measure of the Big-Five personality domains. *Journal of Research in personality*, 37(6), 504-528.')
if st.checkbox('Show more info about the scales'):
    st.write('The ***Generic Conspiracist Beliefs Scale (GCBS)*** items were rated on a 5-point Likert-type scale, with a qualitative label associated with each point')
    st.image('https://www.frontiersin.org/files/Articles/46573/fpsyg-04-00279-HTML/image_m/fpsyg-04-00279-at001.jpg')
    st.write('The ***Ten Item Personality Inventory (TIPI)*** was administered by rating the statement *"I see myself as: _"*')
    st.image('https://us.v-cdn.net/6030293/uploads/editor/mj/z80t41fkdg1m.png')

st.subheader('Data preprocessing')
st.image('https://miro.medium.com/max/4096/1*PnbRC7LG_CutKvrOHpW7kw.png')
st.write('Before starting the whole process of data cleaning and analysis, raw data has been be trasformed in the most convenient way for the upcoming steps.')
st.write('In particular, the following steps has been applied:')
st.write("* dropping useless columns \n * assigning the correct data types \n * labelling categorical data according to the codebook \n * grouping text fields values \n * identifying missing values (according to the codebook) \n * dropping incomplete responses (GCBS and TIPI) \n * computing GCBS and TIPI scores (according to literature). *This step is crucial as it allows us to transform raw data (indicators) into a value (score) that reflects an individual's position on an underlying construct (such as personality or beliefs)* \n * recoding age in groups (according to literature). *This step is important as people in the same group share psychological characteristcs, beliefs and behaviors.* \n * recoding family size in groups. *This step is important as linving in small (i.e. only child) or big families has long-lasting effects on paople's behvaiour and psychological outcomes.* \n * computing variables for survey cleaning (straightliners, speed responses, outliers, validity check errors) \n * renaming columns (with prefixes for better organization)")
if st.checkbox('Show reference on scoring and grouping'):
    st.write('Details on GCBS scoring can be found in:')
    st.write('>Brotherton, Robert, Christopher C. French, and Alan D. Pickering. Measuring belief in conspiracy theories: the generic conspiracist beliefs scale. *Frontiers in psychology* 4 (2013).')
    st.write('Details on how to compute TIPI scores can be found in:')
    st.write('>Gosling, S. D., Rentfrow, P. J., & Swann Jr., W. B. (2003). *Ten Item Personality Measure (TIPI)*. GOZ LAB. Retrieved December 28, 2021, from http://gosling.psy.utexas.edu/scales-weve-developed/ten-item-personality-measure-tipi/')
    st.write('Details on family size effects can be found in:')
    st.write('>*Family size.* Iresearchnet - Psychology Reseach and Reference. (2017, March 14). Retrieved December 28, 2021, from http://psychology.iresearchnet.com/developmental-psychology/family-psychology/family-size/')
    st.write('Details on how to group the variable age according to psychology (each age group has its own characteristics) ca be found in:')
    st.write('>Lally, M., & Valentine-French S. (2019). *Lifespan Development: A Psychological Perspective (2nd ed.).* Retrieved from http://dept.clcillinois.edu/psy/LifespanDevelopment.pdf')

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

############## Web App - Data Cleaning ##############

st.subheader('Data cleaning')
st.image('https://blog.insycle.com/hubfs/The%20Complete%20Guide%20to%20Customer%20Data%20Cleaning.png')
st.write('In order to remove noise and invalid responses, data related to participants who gave quick answers, failed validity checks, or provided straightline responses was removed.')
st.write('The cleaning process removed almost half the individuals (1302 out of 2495) leaving 1193 valid records.')

st.write('**Dataset preview**')
st.write(gcbs_clean_df.head(10))

options = st.multiselect('Show more info about the dataset', ['Type of variables', 'Missing values', 'Quick summary of the values'])
if 'Type of variables' in options:
    st.write('**Type of variables**')
    buffer = io.StringIO()
    gcbs_clean_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
if 'Missing values' in options:
    st.write('**Missing values**')
    st.write(gcbs_clean_df.isnull().sum()[gcbs_clean_df.isnull().sum() > 0])
if 'Quick summary of the values' in options:
    st.write('**Quick summary of quantitative values**')
    st.write(gcbs_clean_df.describe().T)

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')
csv = convert_df(gcbs_clean_df)
st.download_button(label='Download clean data as CSV', data=csv, file_name='gcbs_clean.csv', mime='text/csv')

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

############## Web App - Data Quality ##############

st.subheader('Data quality')
st.image('https://www.grunge.com/img/gallery/the-strange-history-of-phrenology/intro-1596120421.jpg')
st.write('When it comes to psychological data, data quality is a serious business as the quality of the output strongly depends on the ability of the scales to actually measure the psychologial dimension which is intented to be measured. So I computed Cronbach’s alpha for both *GCBS (Conspiracist Belief)* and *TIPI (Personality).*')
st.write('**Cronbach’s alpha** is a measure used to assess the reliability, or internal consistency, of a set of scale or test items. It is computed by correlating the score for each scale item with the total score for each observation (usually individual survey respondents or test takers), and then comparing that to the variance for all individual item scores:')
st.latex(r'''{\alpha} =  (\frac{k}{k-1}) (1- \sum \limits _{i=1} ^{k} \frac{{\sigma}_{y_{i}}^2}{{\sigma}_{x}^2})''')
st.write('where: \n  * ${k}$ refers to the number of scale items \n * ${\sigma}_{y_{i}}^2$ refers to the variance associated with item ${i}$ \n * ${\sigma}_{x}^2$ refers to the variance associated with the observed total scores')
st.write('**Warning:** As pointed out by Eisinga and colleagues (2013), when items=2 (as in *TIPI* subscales) coefficient alpha almost always underestimates true reliability.')
st.write(cronbach_df)
st.write('Cronbach’s alphas show excellent internal consistency for the *Generic Conspiracist Beliefs Scale (GCBS)*, while results from the personality assessment seems questionable. With just 10 items *Personality Scale (TIPI)* struggle to capture internal consistency for subscales. However composite reliability alpha>=0.6 is considered satisfactory for exploratory research (Nunally & Bernstein, 1994 ).')
if st.checkbox('Show reference on data quality'):
    st.write('>Eisinga, Rob; Grotenhuis, Manfred te; Pelzer, Ben (2013). *The reliability of a two-item scale: Pearson, Cronbach, or Spearman-Brown?.* International Journal of Public Health, 58(4), 637–642. doi:10.1007/s00038-012-0416-3')
    st.write('>Nunnally, J.C. and Bernstein, I.H. (1994) *The Assessment of Reliability.* Psychometric Theory, 3, 248-292.')

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
    random_color = tuple(random.randint(1,255) for _ in range(3))
    total_responses = sum(df[col].value_counts())
    fig = df[col].value_counts().sort_index(ascending=True).iplot(kind='bar', title='{} (N={})'.format(title,total_responses), color='rgb{}'.format(str(random_color)), theme='white', asFigure=True)
    return fig

############## Web App - Partecipants ##############

st.subheader('Partecipants')
st.image('https://designmuseumfoundation.org/wp-content/uploads/2021/01/DIA_illustration_cropped.jpg')

col_tile_dict = {
    'Age groups':'DEMO_agegroup',
    'Education':'DEMO_education',
    'Family type':'DEMO_familytype',
    'Area':'DEMO_urban',
    'Gender':'DEMO_gender',
    'Language':'DEMO_engnat',
    'Hand preference':'DEMO_hand',
    'Religion':'DEMO_religion',
    'Sexual orientation':'DEMO_orientation',
    'Racial identification':'DEMO_race',
    'Voted':'DEMO_voted',
    'Marital status':'DEMO_married',
    'College Major':'DEMO_major_cluster'
}

options = st.multiselect('Select which charts you want to visualize', ['Age groups', 'Education', 'Family type','Area','Gender','Language','Hand preference','Religion','Sexual orientation','Racial identification','Voted','Marital status','College Major'])
for variable in options:
    if variable in ['Age groups','Education','Family type']:
        st.write(barChart(gcbs_clean_df, col=col_tile_dict[variable], title=variable))
    else:
        st.write(pieChart(gcbs_clean_df, col=col_tile_dict[variable], title=variable))