from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import os
    
def plot(train_loss, val_loss, name):
    # Load the training and validation loss dictionaries
    train_loss = load(open(train_loss, 'rb'))
    val_loss = load(open(val_loss, 'rb'))
    # Retrieve each dictionary's values
    train_values = train_loss.values()

    val_value1 = []
    val_value2 = []
    val_value3 = []
    for i in range(len(val_loss)):
        val_value1.append(val_loss[i][0])
        val_value2.append(val_loss[i][1])
        val_value3.append(val_loss[i][2])
    

    epochs = range(1, 16)
    
    # # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    plt.plot(epochs, val_value1, label='Validation Loss 1')
    plt.plot(epochs, val_value2, label='Validation Loss 2')
    plt.plot(epochs, val_value3, label='Validation Loss 3')
    
    # # Add in a title and axes labels
    plt.title('Training Loss, Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # # Save the plot
    plt.legend(loc='best')
    plt.savefig(name)
    plt.show()
    plt.close()


plot('./pkl/Task1_resnet50_train_loss_0.01.pkl', './pkl/Task1_resnet50_val_loss_0.01.pkl', 'resnet50_0.01.png')