import torch
from torch.utils.data import Dataset, DataLoader
import torch.cuda as cuda
import matplotlib.pyplot as plt
# Helper functions

#################
# Checkpointing #
#################
def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar', always_save=False):
    """Save checkpoint if a new best is achieved"""
    if is_best or always_save:
        print ("=> Saving checkpoint to: {}".format(filename))
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

####################
# Loss Computation #
####################
def compute_loss(loss, output, data, lam, model):
    out = 0.5*loss(output, data)
    # Extract layer weights for regularization
    for name, param in model.named_parameters():
        if "weight" in name:
            out += lam*(param.norm()**2)
    return out

##############
# Validation #
##############
def validate(loss, model, val_data, lam, weights, cuda_available):
    """Computes the validation error of the model on the validation set.
    val_data should be a Dataset where the first batch loads all of the data."""
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True)
    _, val_tensor = next(enumerate(val_loader))
    if cuda.is_available():
        val_tensor.cuda()
    val_set = Variable(val_tensor, requires_grad = False)
    output = model(val_set)
    return compute_loss(loss, output, val_set, lam, weights)

############
# Plotting #
############
def save_train_val_loss_plots(trainlosses, vallosses, epoch):
    # Train loss
    fig = plt.figure()
    plt.plot(trainlosses)
    plt.title("Train loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("trainloss_epoch{}.png".format(epoch))
    # Train loss
    fig = plt.figure()
    plt.plot(trainlosses)
    plt.title("Val loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.savefig("Val loss{}.png".format(epoch))

