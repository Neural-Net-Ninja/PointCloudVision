import random


validation_losses = {}
dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for dropout in dropout_values:

    # Generate a list of 10 random 2 decimal float numbers
    loss_with_dropout = round(random.uniform(0.01, 10.00), 2)
    loss_no_dropout = round(random.uniform(0.01, 10.00), 2)

    validation_losses[dropout] = (loss_with_dropout, loss_no_dropout)

# Find the dropout value that gives the lowest validation loss
print('validation_losses:____',validation_losses)
losses_with_no_dropout = [loss[1] for loss in validation_losses.values()]
loss_no_dropout = min(losses_with_no_dropout)
for dropout, loss in validation_losses.items():
    if loss[1] == loss_no_dropout:
        loss_with_dropout = loss[0]
        print('Dropout value that gives the lowest validation loss:____', dropout)
        break
