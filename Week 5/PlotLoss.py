from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import os
    
def plot(train_loss, val_loss, val_acc, name):
    # Load the training and validation loss dictionaries
    # train_loss = load(open(train_loss, 'rb'))
    # val_loss = load(open(val_loss, 'rb'))
    # val_acc = load(open(val_acc, 'rb'))
    print(val_acc)
    # Retrieve each dictionary's values
    train_values = train_loss.values()
    val_values = val_loss.values()
    acc_values = val_acc.values()
    print(train_loss)
    # print(val_loss)
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, 16)
    
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    plt.plot(epochs, val_values, label='Validation Loss')
    plt.plot(epochs, acc_values, label='Validation Accuracy')
    
    # Add in a title and axes labels
    plt.title('Training Loss, Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Save the plot
    plt.legend(loc='best')
    plt.savefig(name)
    plt.show()
    plt.close()

def plotall():
    file_paths = os.listdir('./pkl')

    labels = [
    'Setting A (Train Loss 0.001)', 'Setting A (Train Loss 0.01)',
    'Setting A (Val Acc 0.001)', 'Setting A (Val Acc 0.01)',
    'Setting A (Val Loss 0.001)', 'Setting A (Val Loss 0.01)',
    'Setting B (Train Loss 0.001)', 'Setting B (Train Loss 0.01)',
    'Setting B (Val Acc 0.001)', 'Setting B (Val Acc 0.01)',
    'Setting B (Val Loss 0.001)', 'Setting B (Val Loss 0.01)',
    'Setting C2 (Train Loss 0.001)', 'Setting C2 (Train Loss 0.01)',
    'Setting C2 (Val Acc 0.001)', 'Setting C2 (Val Acc 0.01)',
    'Setting C2 (Val Loss 0.001)', 'Setting C2 (Val Loss 0.01)'
    ]

    # Create subplots for train loss, validation loss, and validation accuracy
    fig, axs = plt.subplots(3, figsize=(10, 12))

    for i in range(0, len(file_paths)):
        # Load the data from pickle files
        with open('./pkl/'+file_paths[i], 'rb') as file:
            data = load(file).values()

        # Determine the plot type (train loss, validation loss, or validation accuracy)
        if 'train_loss' in file_paths[i]:
            print('train:' + file_paths[i])
            ax = axs[0]  # Train Loss
        elif 'val_loss' in file_paths[i]:
            print('val_loss:' + file_paths[i])
            ax = axs[1]  # Validation Loss
        else:
            print('val_acc:' + file_paths[i])
            ax = axs[2]  # Validation Accuracy

        # Plot the data
        ax.plot(data, label=labels[i], linewidth=2)
        

    # Customize the plots
    axs[0].set_title('Train Loss')
    axs[1].set_title('Validation Loss')
    axs[2].set_title('Validation Accuracy')

    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout()
    plt.savefig('Comparison_Plot.png')
    plt.close()
plotall()