import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train_test_files/qttnews.train.csv",header=0,index_col=None)
_,df_test = train_test_split(df,test_size=0.2)
df_test.to_csv("data/train_test_files/qttnews.test.csv",header=True,index=False)
