from src.data_loading import load_data
from src.data_preprocessing import data_preprocessing
from src.data_visualization import visualize_label_distribution, multiclass_pca_projection, binary_class_pca_projection

df = load_data()
df = data_preprocessing(df)
print (df.head())

visualize_label_distribution(df)
multiclass_pca_projection(df)
binary_class_pca_projection(df)

