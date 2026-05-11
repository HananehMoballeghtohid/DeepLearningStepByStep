device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1000

# +
losses = []
val_losses = []

for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f'epoch {epoch}/{epochs}')
        
    loss = mini_batch_loop(train_loader, device, train_step_function)
    losses.append(loss)
    
    with torch.no_grad():
        val_loss = mini_batch_loop(val_loader, device, val_step_function)
        val_losses.append(val_loss)
