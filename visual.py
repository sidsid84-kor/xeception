import matplotlib.pyplot as plt

def plot_train_val_loss(num_epochs, loss_history):
    plt.title('Train-Val Loss')
    plt.plot(range(1, num_epochs+1), loss_history['train'], label='train')
    plt.plot(range(1, num_epochs+1), loss_history['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()

def plot_train_val_accuracy(num_epochs, metric_history):
    plt.title('Train-Val Accuracy')
    plt.plot(range(1, num_epochs+1), metric_history['train'], label='train')
    plt.plot(range(1, num_epochs+1), metric_history['val'], label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()