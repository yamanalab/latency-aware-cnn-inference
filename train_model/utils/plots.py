import matplotlib.pyplot as plt


def plot_and_save_training_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history, save_path):
    plt.style.use('ggplot')
    fig, (acc_ax, loss_ax) = plt.subplots(nrows=2, constrained_layout=True)

    acc_ax.plot(train_acc_history)
    acc_ax.plot(val_acc_history)
    acc_ax.set_title('Model Accuracy')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylim(0, 100)
    acc_ax.legend(['training', 'validation'], loc='upper left')
    acc_ax.grid()

    loss_ax.plot(train_loss_history)
    loss_ax.plot(val_loss_history)
    loss_ax.set_title('Model Loss')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylim(0, 5.0)
    loss_ax.legend(['training', 'validation'], loc='upper left')
    loss_ax.grid()

    fig.savefig(save_path, dpi=300)
