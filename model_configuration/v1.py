device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.01

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

loss_function = nn.MSELoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=lr)

# This will be used in the training loop
train_step_function = make_train_step_fn(model, loss_function, optimizer)
