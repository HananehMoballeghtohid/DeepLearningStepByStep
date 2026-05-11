torch.manual_seed(0)

# We don't send the whole training data into GPU (device) because of memory.
X_tensor = torch.as_tensor(X).float()
y_tensor = torch.as_tensor(y).float()

dataset = TensorDataset(X_tensor, y_tensor)

# Perform split
ratio = .8
N = len(dataset)
n_train = int(N * ratio)
n_val = N - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
    dataset = train_data,
    batch_size=128,
    shuffle=True,
)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=128,
)
