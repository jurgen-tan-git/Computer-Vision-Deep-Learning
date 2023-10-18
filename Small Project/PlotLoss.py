from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import os
    
def plot(train_loss, val_loss, name):
    train_loss = load(open(train_loss, 'rb'))
    val_loss = load(open(val_loss, 'rb'))

    # print(train_loss)
    print(val_loss)

    
    # train_values1 = []
    # train_values2 = []
    # train_values3 = []

    # val_value1 = []
    # val_value2 = []
    # val_value3 = []
    # for i in range(15):
    #     train_values1.append(train_loss[i][0])
    #     train_values2.append(train_loss[i][1])
    #     train_values3.append(train_loss[i][2])
    #     val_value1.append(val_loss[i][0])
    #     val_value2.append(val_loss[i][1])
    #     val_value3.append(val_loss[i][2])
    

    # epochs = range(1, 16)
    
    # # # Plot and label the training and validation loss values
    # plt.plot(epochs, train_values1, label='Training Loss - Augmentation 1')
    # plt.plot(epochs, train_values2, label='Training Loss - Augmentation 2')
    # plt.plot(epochs, train_values3, label='Training Loss - Augmentation 3')
    # plt.plot(epochs, val_value1, label='Validation Loss - Augmentation 1')
    # plt.plot(epochs, val_value2, label='Validation Loss - Augmentation 2')
    # plt.plot(epochs, val_value3, label='Validation Loss - Augmentation 3')
    
    # # # Add in a title and axes labels
    # plt.title('Training Loss, Validation Loss and Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    
    # # # Save the plot
    # plt.legend(loc='best')
    # plt.savefig(name)
    # plt.show()
    # plt.close()


plot('./pkl/Task1_resnet50_train_loss_0.01.pkl', './pkl/Task1_resnet50_val_loss_0.01.pkl', 'resnet50_0.01.png')