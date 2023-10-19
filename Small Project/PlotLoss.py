from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import os


def Task1plot(train_loss, val_loss, name):
    train_loss = load(open(train_loss, 'rb'))
    val_loss = load(open(val_loss, 'rb'))
    
    train_values = []

    val_values = []
    
    for i in range(15):
        train_values.append(train_loss[str(i) + 'A' + str(0)]) # 0 is the first Augmentation
        val_values.append(val_loss[str(i) + 'A' + str(0)]) # 0 is the first Augmentation
    

    epochs = range(1, 16)
    plt.plot(epochs, train_values, label='Training Loss')
    plt.plot(epochs, val_values, label='Validation Loss')

    
    # # Add in a title and axes labels
    plt.title('Training Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Save and show the plot
    plt.legend(loc='best')
    plt.savefig(name)
    plt.show()
    plt.close()


def Task2plot(train_loss, val_loss, name):
    train_loss = load(open(train_loss, 'rb'))
    val_loss = load(open(val_loss, 'rb'))
    
    for i in range(len(train_loss)):
        train_loss[i] = train_loss[i].to('cpu').tolist()
        val_loss[i] = val_loss[i].to('cpu').tolist()
    train_loss = train_loss.values()
    val_loss = val_loss.values()



    epochs = range(1, 11)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    
    # # Add in a title and axes labels
    plt.title('Training Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    
    # Save and show the plot
    plt.legend(loc='best')
    plt.savefig(name)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Task1plot('./pkl/Task1_train_loss_0.01.pkl', './pkl/Task1_val_loss_0.01.pkl', 'Task1_0.01.png')
    # Task2plot('./pkl/multilabel-model_train_loss_0.1.pkl', './pkl/multilabel-model_val_loss_0.1.pkl', 'Task2_0.01.png')
