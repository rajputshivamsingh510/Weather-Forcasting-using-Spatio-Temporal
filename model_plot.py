import matplotlib.pyplot as plt

def plot_model_history(histories, lstm, cnnlstm,gru):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['loss'], label=f'{name} Train Loss')
        plt.plot(history.history['val_loss'], label=f'{name} Val Loss')
    plt.title('Loss (Error Rate) Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy if available
    plt.subplot(1, 2, 2)
    acc_available = False
    for history, name in zip(histories, model_names):
        if 'accuracy' in history.history:
            acc_available = True
            plt.plot(history.history['accuracy'], label=f'{name} Train Acc')
            plt.plot(history.history['val_accuracy'], label=f'{name} Val Acc')
    if acc_available:
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Accuracy not available for models', ha='center', va='center', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

