import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Add custom CSS to the Streamlit app
st.markdown("""
<style>
    body {
        background-color: black;
    }
    [data-testid=stSidebar] {
        background-color: #000000;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    "## "

# Define the questions for each personality trait
ext_questions = {
    'E1': 'I am the life of the party.',
    'E2': 'I talk a lot',
    'E3': 'I feel comfortable around people.',
    'E4': "I don't keep in the background.",
    'E5': 'I start conversations.',
    'E6': 'I have a lot to say.',
    'E7': 'I talk to a lot of different people at parties.',
    'E8': 'I like to draw attention to myself.',
    'E9': "I don't mind being the center of attention.",
    'E10': 'I am not shy around strangers.'
}

neu_questions = {
    'N1': 'I get stressed out easily.',
    'N2': 'I am stressed most of the time.',
    'N3': 'I worry about things.',
    'N4': 'I often feel blue.',
    'N5': 'I am easily disturbed.',
    'N6': 'I get upset easily.',
    'N7': 'I change my mood a lot.',
    'N8': 'I have frequent mood swings.',
    'N9': 'I get irritated easily',
    'N10': 'I often feel blue.'
}

agr_questions = {
    'A1': 'I feel concern for others.',
    'A2': 'I am interested in people.',
    'A3': 'I don\'t insult people.',
    'A4': 'I sympathize with others feelings.',
    'A5': 'I am interested in other people problems.',
    'A6': 'I have a soft heart',
    'A7': 'I am quite interested in others.',
    'A8': 'I take time out for others',
    'A9': 'I feel others emotions',
    'A10': 'I make people feel at ease.'
}

con_questions = {
    'C1': 'I am always prepared.',
    'C2': 'I don\'t leave my belongings around.',
    'C3': 'I pay attention to details',
    'C4': 'I don\'t make a mess of things.',
    'C5': 'I get chores done right away.',
    'C6': 'I rarely forget to put things back in their proper place.',
    'C7': 'I like order.',
    'C8': 'I don\'t shirk my duties.',
    'C9': 'I follow a schedule.',
    'C10': 'I am exacting in my work.'
}

opn_questions = {
    'O1': 'I have a rich vocabulary.',
    'O2': 'I easily understand abstract ideas.',
    'O3': 'I have a vivid imagination.',
    'O4': 'I am interested in abstract ideas.',
    'O5': 'I have excellent ideas',
    'O6': 'I have a good imagination.',
    'O7': 'I am quick to understand things.',
    'O8': 'I use difficult words.',
    'O9': 'I spend time reflecting on things.',
    'O10': 'I am full of ideas.'
}

questions = {
    'Extroversion': ext_questions,
    'Neuroticism': neu_questions,
    'Agreeableness': agr_questions,
    'Conscientiousness': con_questions,
    'Openness': opn_questions
}

# Load the data
@st.cache(allow_output_mutation=True)
def load_data():
    return pd.DataFrame(columns=questions.keys())

data1 = load_data()

# Create a Streamlit app
st.title('Personality Traits Analysis')
df = pd.read_csv('data.csv', sep='\t')


# Display plot for a specific personality trait
def display_plot(questions, title, display):
    if display:
        fig, axes = plt.subplots(5, 2, figsize=(15, 25))
        for i, (q, ax) in enumerate(zip(questions.keys(), axes.flatten())):
            sns.countplot(x=q, edgecolor="black", alpha=0.7, data=df, ax=ax)
            ax.set_title(f"Q&As Related to {title} Personality: \n{questions[q]}")
        plt.tight_layout()
        st.pyplot(fig)

# Display and hide buttons for each personality trait
display_status = {trait: False for trait in questions}
for trait in questions:
    if st.sidebar.button(f'{trait} Plot',type="primary"):
        display_status[trait] = not display_status[trait]
    display_plot(questions[trait], trait, display_status[trait])

st.markdown("<style>body {background-color: black;}</style>", unsafe_allow_html=True)
# Display questions and collect responses
for trait in questions:
    st.subheader(f'{trait} Answers')
    for q_key, q_value in questions[trait].items():
        response_map = {'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5}
        response = st.radio(q_value, ('Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'), key=q_key)
        response_integer = response_map[response]  # Map selected option to integer
        data1.loc[0, q_key] = response_integer


# Button to save responses
if st.sidebar.button("Save Responses",type="primary"):
    data1.to_csv('responses.csv', index=False)
    st.success("Responses saved successfully!")

# Button to display responses (in sidebar)
if st.sidebar.button("Display Responses",type="primary"):
    st.subheader("Responses:")
    st.write(data1)

# Apply scaling
changed_qs = ['E2', 'E4', 'E6', 'E8', 'E10',
               'N2', 'N4',
               'A1', 'A3', 'A5', 'A7',
               'C2', 'C4', 'C6', 'C8',
               'O2', 'O4', 'O6']

def invert_and_scale(column):
    df[column].replace(5, -1, inplace=True)
    df[column].replace(4, -0.5, inplace=True)
    df[column].replace(3, 0, inplace=True)
    df[column].replace(2, 0.5, inplace=True)
    df[column].replace(1, 1, inplace=True)

scaler = MinMaxScaler(feature_range=(-1,1))
for col in df.iloc[:, 7:]:
    if col in changed_qs:
        invert_and_scale(col)
    else:
        df[col] = scaler.fit_transform(df[[col]])
df_model = df.drop(['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country'], axis=1)

# The additional part you provided
mid = df.copy()

mid = mid.drop(['race','age','engnat','gender','hand','source','country'],axis = 1)

mid['Social'] = (mid.E1 + mid.E3 + mid.E5 + mid.E7 + mid.E9)/5
mid.drop(['E1','E3','E5','E7','E9'],axis = 1, inplace = True)

mid['Not_Social'] = (mid.E2 + mid.E4 + mid.E6 + mid.E8 + mid.E10)/5
mid.drop(['E2','E4','E6','E8','E10'],axis = 1, inplace = True)

mid['Optimal_Mood'] = (mid.N2 + mid.N4)/2
mid.drop(['N2','N4'],axis = 1, inplace = True)

mid['Disturbed_Mood'] = (mid.N1 + mid.N3 + mid.N5 +mid.N6+ mid.N7 +mid.N8 + mid.N9 + mid.N10)/8
mid.drop(['N1','N3','N5','N6','N7','N8','N9','N10'],axis = 1, inplace = True)

mid['Positive_social_interactions'] = (mid.A2 + mid.A4 + mid.A6 + mid.A8 + mid.A9 + mid.A10)/6
mid.drop(['A2','A4','A6','A8','A9','A10'],axis = 1, inplace = True)

mid['Negative_Social_Interactions'] = (mid.A1 + mid.A3 + mid.A5 + mid.A7)/4
mid.drop(['A1','A3','A5','A7'],axis = 1, inplace = True)

mid['Organised'] = (mid.C1 + mid.C3 + mid.C5 + mid.C7 + mid.C9 + mid.C10)/6
mid.drop(['C1','C3','C5','C7','C9','C10'],axis = 1, inplace = True)

mid['Unorganised'] = (mid.C2 + mid.C4 + mid.C6 + mid.C8)/4
mid.drop(['C2','C4','C6','C8'],axis = 1, inplace = True)

mid['Thinker'] = (mid.O1 + mid.O3 + mid.O5 + mid.O7 + mid.O8 + mid.O9 + mid.O10)/5
mid.drop(['O1','O3','O5','O7','O8','O9','O10'],axis = 1, inplace = True)

mid['Non_Thinker'] = (mid.O2 + mid.O4 + mid.O6)/3
mid.drop(['O2','O4','O6'],axis = 1, inplace = True)

# Function to display heatmap
def display_heatmap():
    corr = mid.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, square=True, fmt='.2f')
    plt.title('Heatmap of the Correlations of the Question variables', fontsize=17)
    return plt.gcf()  # Return the current figure



# Sidebar button to display heatmap
if st.sidebar.button("Display Heatmap",type="primary"):
    st.pyplot(display_heatmap())

# Below is the rest of your code...


kmeans = KMeans(n_clusters=5, random_state=42)
k_fit = kmeans.fit(df_model)

pd.options.display.max_columns = 10
predictions = k_fit.labels_
df_model['clusters'] = predictions

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_model)
    wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(wcss, columns=['wcss'])
wcss = wcss.reset_index()
wcss = wcss.rename(columns={'index': 'clusters'})
wcss['clusters'] += 1

df_model_4 = df.drop(['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country'], axis=1)

kmeans = KMeans(n_clusters=4, random_state=42)
k_fit = kmeans.fit(df_model_4)

predictions = k_fit.labels_
df_model_4['clusters'] = predictions

pca_3 = PCA(n_components=3)
pca_fit = pca_3.fit_transform(df_model_4)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2', 'PCA3'])
df_pca['clusters'] = predictions

px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', template='seaborn',
              title='Personality Clusters after PCA in 3D',
              color='clusters', opacity=0.5, width=650, height=700)

pca_2 = PCA(n_components=2)
pca_fit = pca_2.fit_transform(df_model_4)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['clusters'] = predictions

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='clusters', alpha=0.8)
plt.title('Personality Clusters after PCA in 2D')

df['extraversion_score'] = 0
for x in ext_questions.keys():
    df['extraversion_score'] += df[x]

df['neuroticism_score'] = 0
for x in neu_questions.keys():
    df['neuroticism_score'] += df[x]

df['agreeableness_score'] = 0
for x in agr_questions.keys():
    df['agreeableness_score'] += df[x]

df['conscientiousness_score'] = 0
for x in con_questions.keys():
    df['conscientiousness_score'] += df[x]

df['openness_score'] = 0
for x in opn_questions.keys():
    df['openness_score'] += df[x]
df_model = df[['extraversion_score', 'neuroticism_score', 'agreeableness_score', 'conscientiousness_score', 'openness_score']]

kmeans = KMeans(n_clusters=4, random_state=42)
k_fit = kmeans.fit(df_model)

pd.options.display.max_columns = 10
predictions = k_fit.labels_
df_model['clusters'] = predictions
pca_3 = PCA(n_components=3)
pca_fit = pca_3.fit_transform(df_model)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2', 'PCA3'])
df_pca['clusters'] = predictions
px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', template='seaborn',
              title='Personality Clusters after PCA in 3D',
              color='clusters', opacity=0.5, width=800, height=700)
# Button to display 3D scatter plot
if st.sidebar.button("Display 3D PCA visualization",type="primary"):
    st.plotly_chart(px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', template='seaborn',
                                   title='Personality Clusters after PCA in 3D',
                                   color='clusters', opacity=0.5, width=800, height=700))

pca_2 = PCA(n_components=2)
pca_fit = pca_2.fit_transform(df_model)

# Button to display 2D scatter plot
if st.sidebar.button("Display 2D PCA visualization",type="primary"):
    plt.figure(figsize=(10,10))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='clusters', alpha=0.8)
    plt.title('Personality Clusters after PCA in 2D')
    st.pyplot(plt)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['clusters'] = predictions

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='clusters', alpha=0.8)
plt.title('Personality Clusters after PCA in 2D');

desc = df_model.groupby('clusters')[['extraversion_score', 'neuroticism_score', 'agreeableness_score', 'conscientiousness_score', 'openness_score']].describe()

summary = pd.concat(
    objs=(i.set_index('clusters') for i in (
        desc['extraversion_score'][['count', 'mean']].reset_index(),
        desc['neuroticism_score'][['mean']].reset_index(),
        desc['agreeableness_score'][['mean']].reset_index(),
        desc['conscientiousness_score'][['mean']].reset_index(),
        desc['openness_score'][['mean']].reset_index())),
    axis=1,
    join='inner').reset_index()
summary.columns = ['clusters', 'cluster_count', 'extraversion_mean', 'neuroticism_mean', 'agreeableness_mean', 'conscientiousness_mean', 'openness_mean']

plt.figure(figsize=(9, 7))
summary.plot(x='clusters',
             y=['extraversion_mean', 'neuroticism_mean', 'agreeableness_mean', 'conscientiousness_mean', 'openness_mean'],
             kind='bar',
             ylabel='Mean of characteristics',
             fontsize=14, cmap="tab20b").legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                                labels=['Mean of Extraversion',
                                                        'Mean of Neuroticism',
                                                        'Mean of Agreeableness',
                                                        'Mean of Conscientiousness',
                                                        'Mean of Openness'])
plt.title('The 4 Clusters of BIG5 Personality Test', fontsize=14)
plt.show()

# Apply scaling to user's data
my_data1 = data1
my_data2 = my_data1

def invert_and_scale1(column):
    my_data2[column].replace(5, -1, inplace=True)
    my_data2[column].replace(4, -0.5, inplace=True)
    my_data2[column].replace(3, 0, inplace=True)
    my_data2[column].replace(2, 0.5, inplace=True)
    my_data2[column].replace(1, 1, inplace=True)

for col in my_data1.iloc[:,:]:
    if col in changed_qs:
        invert_and_scale1(col)
    else:
        my_data2[col] = scaler.fit_transform(my_data2[[col]])

# Calculate personality scores for user's data
my_data3 = my_data2.copy()

my_data3['extraversion_score'] = 0
for x in ext_questions.keys():
    my_data3['extraversion_score'] += my_data2[x]

my_data3['neuroticism_score'] = 0
for x in neu_questions.keys():
    my_data3['neuroticism_score'] += my_data2[x]

my_data3['agreeableness_score'] = 0
for x in agr_questions.keys():
    my_data3['agreeableness_score'] += my_data2[x]

my_data3['conscientiousness_score'] = 0
for x in con_questions.keys():
    my_data3['conscientiousness_score'] += my_data2[x]

my_data3['openness_score'] = 0
for x in opn_questions.keys():
    my_data3['openness_score'] += my_data2[x]

my_data4 = my_data3[['extraversion_score', 'neuroticism_score', 'agreeableness_score', 'conscientiousness_score', 'openness_score']]

# Predict personality cluster for user's data
my_personality = k_fit.predict(my_data4)




# Button to display my personality
if st.sidebar.button("Display My Personality",type="primary"):
    st.write("My Personality Cluster: ", my_personality)

    # Filter summary dataframe for the current cluster
    cluster_data = summary[summary['clusters'] == my_personality[0]]

    # Plot the bar chart for the current cluster
    plt.figure(figsize=(9, 7))
    cluster_data.plot(x='clusters',
                      y=['extraversion_mean', 'neuroticism_mean', 'agreeableness_mean', 'conscientiousness_mean',
                         'openness_mean'],
                      kind='bar',
                      ylabel='Mean of characteristics',
                      fontsize=12,
                      cmap="tab20b",
                      edgecolor='black',  # Add edge color to bars
                      linewidth=1,  # Set width of edge line
                      alpha=0.8,  # Adjust transparency of bars
                      rot=0)  # Rotate x-axis labels

    # Add legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               labels=['Mean of Extraversion',
                       'Mean of Neuroticism',
                       'Mean of Agreeableness',
                       'Mean of Conscientiousness',
                       'Mean of Openness'],
               fontsize=12)

    # Set title and adjust font size
    plt.title(f'Cluster {my_personality[0]}', fontsize=14)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    st.pyplot(plt)
    # Display description below in a styled box
    description = ""
    if my_personality[0] == 0:
        description = "This cluster is made of mostly introverted, quite nervous, friendly and creative but neither careless nor organized personalities."
    elif my_personality[0] == 1:
        description = "This cluster is made of mostly extroverted, friendly, organized and curious and also very confident people."
    elif my_personality[0] == 2:
        description = "This cluster can be described as consisting of mid-level confident introverts who are rather friendly, organized and quite creative."
    elif my_personality[0] == 3:
        description = "This cluster consists of mid-level nervous extraverts who are really friendly and creative but neither careless nor organized."

    plt.figure(figsize=(10, 3))
    bbox_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightblue", alpha=0.5)
    plt.text(0.5, 0.5, description, fontsize=14, ha='center', va='center', wrap=True, bbox=bbox_props, color='black')
    plt.axis('off')
    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    st.pyplot(plt)