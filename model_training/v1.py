epochs = 1000

losses = []

for epoch in range(epochs):
    loss = train_step_function(X_train_tensor, y_train_tensor)
    losses.append(loss)
