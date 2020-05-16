# importing the required libraries
import pandas as pd

# reading the input file
df = pd.read_csv('fer2013.csv')
usages=df['Usage'].unique()

# getting the training samples from the dataset
training=df.sample(frac=0.8)
training['Usage']='Training'
df=df.drop(training.index)

#getting the testing samples from the dataset
testing=df
testing['Usage']='PrivateTest'

# getting the validation samples from the dataset
validation=training.sample(frac=0.1)
validation['Usage']='PublicTest'
training=training.drop(validation.index)

# concatenating all the fields
final_df=pd.concat([training,validation,testing])
print(final_df)

# converting the data frame to csv
final_df.to_csv('fer2013.csv',index=False)