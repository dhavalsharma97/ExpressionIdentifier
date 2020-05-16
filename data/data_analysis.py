# importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# reading the input file
df = pd.read_csv('fer2013.csv')
emotions=df['emotion'].unique()

d={}
for e in emotions:
    d[e]=len(df[df['emotion']==e])

# bar graph
"""plt.title('Dataset Analysis')
plt.xlabel('Image Category')
plt.ylabel('Number of Images')
sns.barplot(list(d.keys()),list(d.values()))
plt.show()"""

# pie chart
plt.title('Dataset Analysis')
plt.pie(list(d.values()), labels=list(d.keys()), autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()