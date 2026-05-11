data = pd.read_csv('housing.csv')
low_price_houses = data[data['median_house_value'] < 500000].reset_index()
X = low_price_houses["median_income"]
y = low_price_houses['median_house_value']

X, y = X.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1)
