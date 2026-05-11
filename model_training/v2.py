device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1000

losses = []
for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f'epoch {epoch}/{epochs}')
        
    batch_losses = []
    for x_batch, y_batch in train_loader:
        # send batch to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        batch_loss = train_step_function(x_batch, y_batch)
        batch_losses.append(batch_loss)
        
    loss = np.mean(batch_losses)
    losses.append(loss)
