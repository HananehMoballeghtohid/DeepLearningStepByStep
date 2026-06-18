lr = 0.01

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(1, 1))

loss_function = nn.MSELoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=lr)
