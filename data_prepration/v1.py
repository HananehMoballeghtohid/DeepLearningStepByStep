
# We don't send the whole training data into GPU (device) because of memory.
X_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

train_data = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    dataset = train_data,
    batch_size=128,
    shuffle=True,
)
