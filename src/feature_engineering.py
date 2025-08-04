def preprocess(df):
    df['tool_condition_binary'] = df['tool_condition'].map({'unworn': 0, 'worn': 1})
    features = df[['feedrate', 'clamp_pressure']]
    target = df['tool_condition_binary']
    return features, target
