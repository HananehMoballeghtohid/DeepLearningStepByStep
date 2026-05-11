epochs = 1000
for epoch in range(epochs):
    model.train()
    
    y_hat = model(X_train_tensor)
    
    loss = loss_function(y_hat, y_train_tensor)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
