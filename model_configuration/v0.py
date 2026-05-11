device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_function = nn.MSELoss(reduction='mean')
