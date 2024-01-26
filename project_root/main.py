from src.data_loading import load_data
from src.data_preprocessing import data_preprocessing

df = load_data()
df = data_preprocessing(df)
print (df.head())