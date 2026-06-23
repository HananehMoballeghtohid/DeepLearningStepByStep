# download and load dataset
train_dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
val_dataset = MNIST(root="./data", train=False, download=True, transform=ToTensor())

# separate images and labels
x_train_tensor = train_dataset.data.unsqueeze(1).float() / 255
y_train = train_dataset.targets
y_train_tensor = (y_train == 2).float().unsqueeze(1)

x_val_tensor = val_dataset.data.unsqueeze(1).float() / 255
y_val = val_dataset.targets
y_val_tensor = (y_val == 2).float().unsqueeze(1)

train_composer = Compose([RandomHorizontalFlip(p=.5),
                        Normalize(mean=(.5,), std=(.5,))])

val_composer = Compose([Normalize(mean=(.5,), std=(.5,))])

train_dataset = TransformedTensorDataset(
x_train_tensor, y_train_tensor, transform=train_composer
)
val_dataset = TransformedTensorDataset(
x_val_tensor, y_val_tensor, transform=val_composer
)

weighted_random_sampler = make_balanced_sampler(y_train_tensor)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    sampler=weighted_random_sampler,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=128,
)