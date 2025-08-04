from src.data_loader import load_train_data
from src.feature_engineering import preprocess
from src.model import train_model

df = load_train_data()
X, y = preprocess(df)
train_model(X, y)
