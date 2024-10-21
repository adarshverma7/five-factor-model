# Five-Factor Model Personality Prediction
This repository contains a machine learning project designed to predict personality types based on the Big Five Personality Traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) using a dataset of 19,719 responses from Open Psychometrics.

![image](https://github.com/user-attachments/assets/f2d36117-0f26-482d-b90a-091e07deffea)

# Introduction
Personality analysis has always been a crucial aspect of understanding human behavior. The Big Five Personality Traits Model, also known as the Five-Factor Model (FFM), helps describe individuals’ personality using five dimensions:

Openness to Experience: Inventive/curious vs. consistent/cautious.
Conscientiousness: Efficient/organized vs. easy-going/careless.
Extraversion: Outgoing/energetic vs. solitary/reserved.
Agreeableness: Friendly/compassionate vs. challenging/detached.
Neuroticism: Sensitive/nervous vs. secure/confident.

In this project, we leverage machine learning techniques to predict a person’s personality type based on these traits, utilizing a dataset from Open Psychometrics.

# Big Five Personality Traits
Trait and	Description
Openness:	Creativity, curiosity, and willingness to try new things.
Conscientiousness:	Organization, dependability, and goal-oriented behavior.
Extraversion:	Sociability, talkativeness, and assertiveness.
Agreeableness:	Compassion, cooperativeness, and trust in others.
Neuroticism:	Emotional instability, anxiety, and vulnerability.

# Dataset
The dataset used for this project contains 19,719 responses to an online personality questionnaire. This questionnaire was part of a study conducted by Open Psychometrics and provides insights into users' responses to a variety of personality-related questions.

Number of Features: 50
Number of Responses: 19,719
Source: Open Psychometrics

This data was collected (c. 2012) through on interactive online personality test. Participants were informed that their responses would be recorded and used for research at the begining of the test and asked to confirm their consent at the end of the test.

The following items were rated on a five point scale where 1=Disagree, 3=Neutral, 5=Agree (0=missed). All were presented on one page in the order E1, N2, A1, C1, O1, E2...... 

E1	I am the life of the party.
E2	I don't talk a lot.
E3	I feel comfortable around people.
E4	I keep in the background.
E5	I start conversations.
E6	I have little to say.
E7	I talk to a lot of different people at parties.
E8	I don't like to draw attention to myself.
E9	I don't mind being the center of attention.
E10	I am quiet around strangers.
N1	I get stressed out easily.
N2	I am relaxed most of the time.
N3	I worry about things.
N4	I seldom feel blue.
N5	I am easily disturbed.
N6	I get upset easily.
N7	I change my mood a lot.
N8	I have frequent mood swings.
N9	I get irritated easily.
N10	I often feel blue.
A1	I feel little concern for others.
A2	I am interested in people.
A3	I insult people.
A4	I sympathize with others' feelings.
A5	I am not interested in other people's problems.
A6	I have a soft heart.
A7	I am not really interested in others.
A8	I take time out for others.
A9	I feel others' emotions.
A10	I make people feel at ease.
C1	I am always prepared.
C2	I leave my belongings around.
C3	I pay attention to details.
C4	I make a mess of things.
C5	I get chores done right away.
C6	I often forget to put things back in their proper place.
C7	I like order.
C8	I shirk my duties.
C9	I follow a schedule.
C10	I am exacting in my work.
O1	I have a rich vocabulary.
O2	I have difficulty understanding abstract ideas.
O3	I have a vivid imagination.
O4	I am not interested in abstract ideas.
O5	I have excellent ideas.
O6	I do not have a good imagination.
O7	I am quick to understand things.
O8	I use difficult words.
O9	I spend time reflecting on things.
O10	I am full of ideas.

![image](https://github.com/user-attachments/assets/fd61f6b8-670b-49c8-a742-bb3744e1a47e)


# Project Objective
The project aims to analyze the data using machine learning techniques and provide the user with their personality type. The focus is to enhance accuracy in predicting personality traits using advanced algorithms, improving the precision over existing models.

# Solution Strategy
Data Preprocessing: The raw data was cleaned and preprocessed, including dealing with missing values and scaling the data.
Exploratory Data Analysis (EDA): Performed to identify trends, patterns, and outliers in the data. Visualization techniques such as histograms and bar charts were employed to understand the distribution of responses.
Feature Engineering: Handled issues with opposing questions (e.g., introverts and extroverts scoring similarly due to inverted question pairs) by transforming and summing scores to provide distinct personality trait values.
Modeling: Applied K-Means clustering and trained multiple models such as Random Forest, MLP, and XGBoost to predict clusters based on personality scores.
Evaluation: Performed evaluations using accuracy, precision, and recall metrics. The Multilayer Perceptron (MLP) achieved the best performance with 97.3% accuracy.

![image](https://github.com/user-attachments/assets/2f8e2e75-0fd2-4a0f-a973-81975beead09)


# Key Models Used
K-Means Clustering for feature clustering.
Principal Component Analysis (PCA) to visualize data in 2D.
Random Forest, MLP, XGBoost, and others for personality trait predictions.

Before Feature Engineering
![image](https://github.com/user-attachments/assets/7ed34fdd-fb6a-4fef-bad5-16cc47f75004)

After Feature Engineering
![image](https://github.com/user-attachments/assets/c3ad148f-5719-4e0f-bab3-dfe00df321ed)

# Installation
Clone the repository and install the necessary dependencies:

git clone https://github.com/yourusername/five-factor-model.git
cd five-factor-model
pip install -r requirements.txt

# Usage
Run the following commands to predict personality types:
python main.py

Or use the provided Jupyter notebooks for detailed analysis:
jupyter notebook analysis.ipynb

# Results
The best performing model (MLP) achieved 97.3% accuracy. Below are the performances of various models:

Model	Accuracy
Random Forest	94.5%
MLP	97.3%
XGBoost	95.1%
Naive Bayes	85.7%

# Future Improvements
Implement additional features such as personality-based music recommendation systems.
Incorporate deep learning for enhanced model accuracy.
Extend the project to include real-time personality analysis through an interactive web interface (e.g., using Streamlit).

# References
Tejas Bakade et al. (2022), "Big Five Personality Prediction Using Machine Learning Algorithms," Mathematical Statistician and Engineering Applications.

Chi J., & Chi Y. N. (2023), "Cluster Analysis of Personality Types Using Respondents’ Big Five Personality Traits," International Journal of Data Science.

