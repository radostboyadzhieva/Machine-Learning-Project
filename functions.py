import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(file):
    df = pd.read_csv(file)

    Y = df['ClaimNb']

    df['VehIsregular'] = (df['VehGas'] == 'Regular').astype(int)

    df = df.drop(columns= ['VehGas', 'IDpol', 'ClaimNb'])

    def letter_to_index(letter):
        return ord(letter.upper()) - ord('A')

    df['Area'] = df['Area'].apply(letter_to_index)

    df['Exposure'] = df['Exposure'].apply(lambda x: x - 1 if x > 1 else x)

    df = pd.get_dummies(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(df)
   
    return X, Y

def split_data(X, Y, test_size):
    return train_test_split(X, Y, test_size=test_size, random_state=1)