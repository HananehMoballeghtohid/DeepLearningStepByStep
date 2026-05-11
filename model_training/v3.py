device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1000

losses = []
for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f'epoch {epoch}/{epochs}')
        
    loss = mini_batch_loop(train_loader, device, train_step_function)
    losses.append(loss)
