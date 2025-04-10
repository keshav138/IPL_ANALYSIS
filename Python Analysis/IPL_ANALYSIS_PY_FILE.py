
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import stats
from scipy.stats import chi2_contingency
from matplotlib import rcParams
 
rcParams['font.family']='Inter'
rcParams['font.family'] = 'sans-serif'

sns.set_style('whitegrid')
sns.set_context('notebook')

team_colors = {
    'Mumbai Indians': '#045093',
    'Chennai Super Kings': '#F1D01A',
    'Royal Challengers Bangalore': '#DA1818',
    'Kolkata Knight Riders': '#3A225D',
    'Rajasthan Royals': '#254AA5',
    'Sunrisers Hyderabad': '#F26522',
    'Kings XI Punjab': '#D71920',
    'Punjab Kings': '#D71920',
    'Delhi Capitals': '#17449B',
    'Delhi Daredevils': '#ED1B24',
    'Gujarat Titans': '#1C2C3C',
    'Lucknow Super Giants': '#2D8CFF',
    'Gujarat Lions': '#F26622',
    'Deccan Chargers': '#A2AAAD',
    'Rising Pune Supergiant': '#651967',
    'Rising Pune Supergiants': '#651967',
    'Pune Warriors': '#0D97A3',
    'Kochi Tuskers Kerala': '#FF9933',
    'Royal Challengers Bengaluru': '#DA1818',  # Same as RCB
    'Draw': '#999999'
}


# %%
file_path = "IPL_DATASET.xlsx"

batting_df = pd.read_excel(file_path, sheet_name="BATTING DATA")
bowling_df = pd.read_excel(file_path, sheet_name="BOWLING DATA")
match_results_df = pd.read_excel(file_path, sheet_name="MATCH RESULTS")

# %%
print("Batting Data : ")
print(batting_df.info())
print(batting_df.head())

# %%
print("Bowling Data : ")
print(bowling_df.info())
print(bowling_df.head())

# %%
print("Match Results : ")
print(match_results_df.info())
print(match_results_df.head())

# %%
# Confirming missing values in the sheets

print("Missing values in the batting data : ")
print(batting_df.isnull().sum())
print()

print('Missing values in the bowling data : ')
print(bowling_df.isnull().sum())
print()

print("Missing values in the match results data : ")
print(match_results_df.isnull().sum())
print()

# %%
# Searching for duplicates in the data

print("Duplicates in Batting Data:", batting_df.duplicated().sum())
print("Duplicates in Bowling Data:", bowling_df.duplicated().sum())
print("Duplicates in Match Data:", match_results_df.duplicated().sum())

# %%
# Normalizing Column Names

batting_df.columns = batting_df.columns.str.strip().str.replace(" ","_")
bowling_df.columns = bowling_df.columns.str.strip().str.replace(" ","_")
match_results_df.columns = match_results_df.columns.str.strip().str.replace(" ","_")

# %%
# Feature Engineering
# Adding Columns of signifance for further objectives

batting_df['Strike_Rate'] = np.where(
    batting_df['Balls_Faced'] > 0,
    (batting_df['Runs'] / batting_df['Balls_Faced']) * 100,
    0
)


# %%
bowling_df['Economy'] = np.where(
    bowling_df['Balls_Bowled'] > 0, 
    (bowling_df['Runs_Conceded'] / (bowling_df['Balls_Bowled'] / 6)),
    0
)


bowling_df['Bowling_Strike_Rate'] = np.where(
    bowling_df['Dismissals'] > 0,
    (bowling_df['Balls_Bowled'] / bowling_df['Dismissals']),
    np.nan
)


# %%
bowling_df

# %%
# General Trends

batting_integers = batting_df[['Runs','Balls_Faced','Strike_Rate']]
bowling_integers = bowling_df[['Runs_Conceded','Balls_Bowled','Economy','Bowling_Strike_Rate']]


print(bowling_integers.describe(),"\n")
print(bowling_integers.describe())

# %%
batting_corr = batting_integers.corr()
print(f"Batting Correlation:\n {batting_corr}")
print("---------------------")
bowling_corr = bowling_integers.corr()
print(f"Bowling Correlation:\n {bowling_corr}")

# %%
sns.heatmap(batting_corr , annot = True , cmap = "rocket" ,fmt=".2f",linewidths=0.5, linecolor='white',cbar_kws={'shrink':0.8} )
plt.title("Batting Correlation Heatmap : ",fontsize=16, fontweight='bold', pad=20 , loc='right')
plt.show()



# %%
sns.heatmap(bowling_corr , annot = True,  cmap = 'rocket',fmt=".2f",linewidths=0.5, linecolor='white',cbar_kws={'shrink':0.8} )
plt.title("Bowling Correlation HeatMap : ",fontsize=16, fontweight='bold', pad=20)
plt.show()

# %%
## outlier detection

# sns.set(style = "whitegrid")

# plt.figure(figsize = (15,5))

# for i , col in enumerate(batting_integers , 1):
#     plt.subplot(1,3,i)
#     sns.boxplot(x = batting_df[col],color = "skyblue")
#     plt.title(f"Boxplot of {col}")

# plt.tight_layout()
# plt.show()

# %%
# Bowling Boxplots

# plt.figure(figsize=(18, 4))
# for i, col in enumerate(bowling_integers, 1):
#     plt.subplot(1, 4, i)
#     sns.boxplot(x=bowling_df[col], color="lightcoral")
#     plt.title(f'Boxplot of {col}')
# plt.tight_layout()
# plt.show()


# %%
#Objectivessss

# %%
top_run_scorers = (
    batting_df.groupby('Batter',as_index = False)
    .agg({"Runs":"sum"})
    .sort_values("Runs",ascending = False)
)

top_ten_run_scorers = top_run_scorers.head(10)



plt.figure(figsize = (10,6))

barplot = sns.barplot(data = top_ten_run_scorers , y = "Batter", x = "Runs",palette ='crest', hue='Runs' )
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=8)


plt.title("TOP 10 RUNS SCORERS IN IPL (ALL SEASONS)",fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Total Runs",fontsize=16, fontweight='bold', labelpad=20)
plt.ylabel("Battters",fontsize=16, fontweight='bold', labelpad=20)
plt.tight_layout()
plt.show()

# %%
## OBJECTIVE

career_stats = (
    batting_df.groupby("Batter")
    .agg({
        'Runs':'sum',
        'Balls_Faced':'sum'
    })
)

career_stats = career_stats[career_stats['Balls_Faced'] >=  500]
career_stats['Strike_Rate'] = (career_stats['Runs'] / career_stats['Balls_Faced']) * 100

career_strike_rate_top = career_stats.sort_values('Strike_Rate',ascending = False).head(20)


plt.figure(figsize=(10,6))
barplot = sns.barplot(
    data = career_strike_rate_top,
    x = 'Strike_Rate',
    y = 'Batter',
    palette = 'magma',
    hue='Strike_Rate'
)

for container in barplot.containers:
    barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=9)

plt.title("Top 10 Batters by Strike Rate (Min 500 Balls Faced)",fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Strike Rate",fontsize=16, fontweight='bold', labelpad=10)
plt.ylabel("Batter",fontsize=16, fontweight='bold', labelpad=10)
# plt.tight_layout()
plt.show()

# %%
## objective

bowler_stats = (
    bowling_df.groupby('Bowler', as_index=False)
    .agg({
        'Runs_Conceded':'sum',
        'Balls_Bowled' : 'sum'
    })
)

bowler_stats = bowler_stats[bowler_stats['Balls_Bowled'] >= 60]

bowler_stats['Economy'] = (bowler_stats['Runs_Conceded'] / (bowler_stats['Balls_Bowled'] / 6))


reliable = bowler_stats[(bowler_stats['Balls_Bowled']>=500) & (bowler_stats['Economy']<=7.0)]

plt.figure(figsize = (10,6))
sns.scatterplot(
    data = bowler_stats.sort_values("Economy",ascending=True),
    x = 'Balls_Bowled',
    y = 'Economy',
    hue = 'Economy',
    palette = 'coolwarm',
    size = 'Economy',
    sizes = (40,150),
    edgecolor='black',
    legend=False
)

sns.scatterplot(
    data = reliable,
    x = 'Balls_Bowled',
    y = 'Economy',
    color='black',
    s=100,
    marker="*",
    label='Reliable Bowler'
)


plt.title("Economy Rates vs Balls Bowled",fontsize=16, fontweight='bold',pad=20)
plt.xlabel("Total Balls Bowled",fontsize=12, fontweight='bold',labelpad=10)
plt.ylabel("Economy Rate",fontsize=12, fontweight='bold',labelpad=10)
plt.legend([],[],frameon = False)
plt.show()


# %%
# Objective

reliable_bowler = bowler_stats[bowler_stats['Balls_Bowled'] >= 500].head(10)
most_economical_bowlers = reliable_bowler.sort_values("Economy",ascending=True)


plt.figure(figsize=(10,6))


barplot = sns.barplot(
    data = most_economical_bowlers,
    x = 'Economy',
    y = 'Bowler',
    palette = 'magma',
    hue = 'Economy'
)
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=8)

plt.title("BOWLERS WITH THE BEST ECONOMY RATES OVER 500 BALLS",fontweight='bold',pad=20)
plt.xlabel("Economy",fontweight='bold',labelpad=10)
plt.ylabel("Bowler",fontweight='bold',labelpad=10)
# plt.tight_layout()
plt.show()

# %%
## objective

valid_dismissals = ['Bowled','Caught','LBW','Stumped','Run_Out']
dismissal_by_type = bowling_df.groupby('Bowler')[valid_dismissals].sum().sort_values(by = 'Bowled',ascending = False)

for dtype in dismissal_by_type:
    print(f"Top Bowlers by {dtype}")
    print(dismissal_by_type.sort_values(by=dtype,ascending = False)[[dtype]].head(10))
    print()

# %%
dismissal_by_type['Total'] = dismissal_by_type.sum(axis = 1)
top_bowlers = dismissal_by_type.sort_values('Total',ascending = False).head(10)

stacked_df = top_bowlers[valid_dismissals]
stackplot = stacked_df.plot(kind = 'barh', stacked=True, figsize=(12,8), colormap='rocket')

for container in stackplot.containers:
    stackplot.bar_label(container, fmt='%.0f', label_type='center', padding=3, fontsize=8,color='white')



plt.title("Top Bowlers by Total Dismissals (Stacked By Type)",fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Wickets",fontsize=12, fontweight='bold', labelpad=20)
plt.ylabel("Bowlers",fontsize=12, fontweight='bold', labelpad=20)
plt.legend(title='Dismissal Type', bbox_to_anchor = (1.05,1), loc='upper left')
plt.show()

# %%

# Set up plot style
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(24, 13))
axes = axes.flatten()

# Create bar plots for each dismissal type
for i, dismissal in enumerate(dismissal_by_type):
    barplot = sns.barplot(
        x=top_bowlers[dismissal].values,
        y=top_bowlers.index,
        ax=axes[i],
        palette="viridis",
        hue = top_bowlers[dismissal].values
    )
    
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=10)
    
    axes[i].set_title(f"Top Bowlers by {dismissal}",)
    axes[i].set_xlabel("Wickets")
    axes[i].set_ylabel("")

plt.tight_layout()
plt.show()


# %%
## consistent performers across seasons
batting_df['Season'] = batting_df['Season'].astype(int)
bowling_df['Season'] = bowling_df['Season'].astype(int)


batting_filtered = batting_df[(batting_df['Season'] >= 2015) & (batting_df['Season'] <= 2024)]
bowling_filtered = bowling_df[(bowling_df['Season'] >= 2015) & (bowling_df['Season'] <= 2024)]

batting_consistency = batting_filtered.groupby(["Batter","Season"])['Runs'].sum().reset_index()

batting_matrix = batting_consistency.pivot(index='Batter',columns='Season',values='Runs').fillna(0)

batting_matrix['Total_Runs'] = batting_matrix.sum(axis = 1)
consistent_batter_last_ten_years = batting_matrix.sort_values("Total_Runs",ascending=False).head(10)
print(consistent_batter_last_ten_years)


# %%
#consistent bowlers
bowling_consistency = bowling_filtered.groupby(['Bowler','Season'])['Dismissals'].sum().reset_index()

bowling_matrix = bowling_consistency.pivot(index = 'Bowler',columns='Season',values='Dismissals')
bowling_matrix['Total_Wickets'] = bowling_matrix.sum(axis=1)

consistent_bowler_last_ten_years = bowling_matrix.sort_values("Total_Wickets",ascending=False).head(10)
print(consistent_bowler_last_ten_years)


# %%
# heatmap for consistent performers
#batters 

plt.figure(figsize=(15,7))

sns.heatmap(
    consistent_batter_last_ten_years.drop(columns='Total_Runs'),
    cmap='YlOrRd',
    annot=True,
    fmt='.0f',
    linewidths=0.5,
    linecolor='grey',
    annot_kws={'fontsize':9},
    cbar_kws={'label':'Total Runs'}
    )

plt.title("Top Batter - Runs per Season (2015 - 2024)",fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Season',fontsize=12, fontweight='bold', labelpad=20)
plt.ylabel('Batter',fontsize=12, fontweight='bold', labelpad=20)
plt.xticks(rotation=45)
plt.show()

# %%
# heatmap for consistent performers
# bowlers

plt.figure(figsize=(15,7))

sns.heatmap(
    consistent_bowler_last_ten_years.drop(columns='Total_Wickets'),
    cmap='mako_r',
    annot=True,
    fmt='.0f',
    linewidths=0.5,
    linecolor='grey',
    annot_kws={'fontsize':9},
    cbar_kws={'label':'Total Runs'}
    
    )

plt.title("Top Bowler - Dismissals per Season (2015 - 2024)",fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Season',fontsize=12, fontweight='bold', labelpad=20)
plt.ylabel('Bolwer',fontsize=12, fontweight='bold', labelpad=20)
plt.show()

# %%
# toss decision

toss_decision_counts = match_results_df['Toss_Decision'].value_counts()
print(toss_decision_counts)
labels = toss_decision_counts.index.tolist()


plt.figure(figsize=(6,6))
plt.pie(
    toss_decision_counts.values,
    labels = labels,
    autopct='%1.1f%%',
    colors=['#045093', '#F7C948'],
    startangle=90,
    textprops={'fontsize':12,'fontweight':'bold','color':'white'}
)
plt.title("Toss Decision : Bat Vs Field",fontweight='bold',pad=20,fontsize=16)
plt.show()

# %%
match_results_df['Toss_Win_Match_Win'] = np.where(
    match_results_df['Toss_Winner'] == match_results_df['Match_Winner'],
    'Yes','No'
)

toss_match_outcome = match_results_df['Toss_Win_Match_Win'].value_counts()
print(toss_match_outcome)

countplot = sns.countplot(data = match_results_df, x ='Toss_Win_Match_Win',palette='rocket_r',hue=match_results_df['Toss_Win_Match_Win'])

for container in countplot.containers:
    countplot.bar_label(container, fmt='%.0f', label_type='center', padding=3, fontsize=18,color='white')


plt.title("Did Toss Winner also win the match?",fontweight='bold',fontsize=16,pad=20)
plt.xlabel("Toss Winner = Match Winner",labelpad=10)
plt.ylabel("Match Count",labelpad=10)
plt.show()

# %%
# analysis of whether statistically there is a significant relationship between toss outcome and match outcome

null_hypothesis = 'Toss Decision and Match Result are independant.'
alternate_hypothesis = 'Toss Decision affects match result'


contingency_table = pd.crosstab(
    match_results_df['Toss_Decision'],
    match_results_df['Toss_Win_Match_Win']
)

print("Contingency Table: ")
print(contingency_table)

chi2, p , dof , expected = chi2_contingency(contingency_table)

print("\nChi-Square Statistic: ",chi2)
print("Degrees of Freedom: ",dof)
print("P-Value: ",p)

alpha = 0.5
if p < alpha:
    print('\nReject Null Hypothesis : Toss Decision has a statistically significant effect on winning.')
else:
    print("\nFail to reject the Null Hypothesis: Toss Decision does NOT significantly affect match outcome.")

# %%
# Filter matches where toss winner also won the match
toss_win_df = match_results_df[match_results_df['Toss_Winner'] == match_results_df['Match_Winner']]

# using below statement will give toss win picks for all cases
# toss_win_df = match_results_df

venue_decision_win = toss_win_df.groupby(['Venue', 'Toss_Decision'])['Match_ID'].count().reset_index()


venue_decision_pivot = venue_decision_win.pivot(index='Venue', columns='Toss_Decision', values='Match_ID').fillna(0)

venue_decision_pivot['Total'] = venue_decision_pivot.sum(axis=1)
venue_decision_pivot = venue_decision_pivot.sort_values('Total', ascending=False)
venue_decision_pivot.drop(columns='Total', inplace=True)


ax = venue_decision_pivot.plot(
    kind = 'barh',
    figsize=(10,16),
    color={'Bat': '#F7C948', 'Field': '#045093'}
)
ax.invert_yaxis()
 
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=6)
    


plt.title("Toss Decision Leading to Wins by Venue",fontweight="bold",fontsize=16,pad=20,loc='left')
plt.xlabel("Number of Wins (Where Toss Winner Won)",fontweight='bold',fontsize=12,labelpad=10)
plt.ylabel("Venue",fontweight='bold',fontsize=12,labelpad=10,loc='center')
plt.show()

# %%
#player of the match
pom_count = match_results_df['Player_of_Match'].value_counts().reset_index()
pom_count.columns = ['Player','Awards']

top_pom = pom_count.head(10)
print(top_pom)

# %%
plt.figure(figsize=(10,6))
barplot = sns.barplot(data = top_pom, x = 'Awards', y='Player',palette="magma", hue='Awards')

for container in barplot.containers:
    barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=10)


plt.title("Top 10 Players of Man of the Match Awards",fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Number of Awards",fontsize=12, fontweight='bold', labelpad=20)
plt.ylabel("Batters",fontsize=12, fontweight='bold', labelpad=20)
plt.show()

# %%
# analysis by team performance for batting and bowling over the years

team_batting = (
    batting_df.groupby(['Team'])['Runs']
    .sum()
    .reset_index()
    .sort_values(by=['Runs'],ascending=[False])
    .head(10)
)

# team_batting_pivot = team_batting.pivot(index='Team',values='Runs').fillna(0)


plt.figure(figsize=(14,9))
barplot = sns.barplot(data = team_batting, x = 'Team', y='Runs',palette="inferno_r", hue='Runs',orient='x',width=0.5)

for container in barplot.containers:
    barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=10)



plt.title("Total Team Runs All Season",fontsize=16, fontweight='bold', pad=20
)
plt.ylabel("Runs Scored",fontsize=12, fontweight='bold', labelpad=20
)
plt.xlabel("Teams",fontsize=12, fontweight='bold', labelpad=20
)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout()
plt.show()
# team_batting

# %%
# analysis by team performance for batting and bowling over the years

team_bowling = (
    bowling_df.groupby(['Bowling_Team'])['Dismissals']
    .sum()
    .reset_index()
    .sort_values(by=['Dismissals'],ascending=[False])
    .head(10)
)

# team_batting_pivot = team_batting.pivot(index='Team',values='Runs').fillna(0)


plt.figure(figsize=(14,9))
barplot = sns.barplot(data = team_bowling, x = 'Bowling_Team', y='Dismissals',palette="rocket", hue='Dismissals',width=0.5)


for container in barplot.containers:
    barplot.bar_label(container, fmt='%.0f', label_type='edge', padding=3, fontsize=10)



plt.title("Total Team Dismissals All Season",fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Total Dismissals",fontsize=12, fontweight='bold', labelpad=10)
plt.xlabel("Season",loc='left',fontsize=12, fontweight='bold', labelpad=10)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout()
plt.show()
#

# %%



