data = pd.read_csv('housing.csv')
low_price_houses = data[data['median_house_value'] < 500000].reset_index()
X = low_price_houses["median_income"]
y = low_price_houses['median_house_value']

N = len(X)
idx = np.arange(N)
np.random.shuffle(idx)

train_idx = idx[:int(N*0.8)]
val_idx = idx[int(N*0.8):]

X_train, y_train = X[train_idx].to_numpy().reshape(-1, 1), y[train_idx].to_numpy().reshape(-1, 1)
X_val, y_val = X[val_idx].to_numpy().reshape(-1, 1), y[val_idx].to_numpy().reshape(-1, 1)
