import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'models/rf_model.pkl')

def load_model(path='models/rf_model.pkl'):
    return joblib.load(path)

def predict(model, input_df):
    return model.predict(input_df)
