# %%
import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
import torchmetrics
import torchvision.transforms as transforms
# from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import datasets

sys.path.append(os.path.abspath("../utils"))
import activations
from activations import ACTIVATIONS
from ghost_module import GhostModule
from plots import plot_and_save_training_history
from torch_hdf5 import save_params_as_hdf5
from torch_json import save_structure_as_json

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
INPUT_C = 1
INPUT_H = 28
INPUT_W = 28

# %%
train_acc, test_acc = torchmetrics.Accuracy(), torchmetrics.Accuracy()
max_test_acc = 0.0
sw = SummaryWriter(log_dir="./histories")

running_loss_history = []
running_acc_history = []
val_running_loss_history = []
val_running_acc_history = []

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--ghost', action='store_true')
parser.add_argument('--bn', action='store_true')
parser.add_argument('--do', action='store_true')
parser.add_argument('--gap', action='store_true')
parser.add_argument('--act', required=True, choices=ACTIVATIONS)
parser.add_argument('--mode', default='train', choices=['train', 'prune', 'test'])
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()
use_ghost_module = args.ghost
bn_enabled = args.bn  # BatchNormalization flag
do_enabled = args.do  # Dropout flag
gap_enabled = args.gap  # GlobalAveragePooling flag
act_str = args.act
EPOCHS = args.epochs

if act_str == 'relu':
    activation = nn.ReLU
elif act_str == 'swish':
    activation = activations.Swish
elif act_str == 'mish':
    activation = nn.Mish
elif act_str == 'square':
    activation = activations.Square
# ReLU approx.
elif act_str == 'relu_rg4_deg2':
    activation = activations.ReluRg4Deg2
elif act_str == 'relu_rg5_deg2':
    activation = activations.ReluRg5Deg2
elif act_str == 'relu_rg6_deg2':
    activation = activations.ReluRg6Deg2
elif act_str == 'relu_rg7_deg2':
    activation = activations.ReluRg7Deg2
elif act_str == 'relu_rg4_deg4':
    activation = activations.ReluRg4Deg4
elif act_str == 'relu_rg5_deg4':
    activation = activations.ReluRg5Deg4
elif act_str == 'relu_rg6_deg4':
    activation = activations.ReluRg6Deg4
elif act_str == 'relu_rg7_deg4':
    activation = activations.ReluRg7Deg4
# Swish approx.
elif act_str == 'swish_rg4_deg2':
    activation = activations.SwishRg4Deg2
elif act_str == 'swish_rg5_deg2':
    activation = activations.SwishRg5Deg2
elif act_str == 'swish_rg6_deg2':
    activation = activations.SwishRg6Deg2
elif act_str == 'swish_rg7_deg2':
    activation = activations.SwishRg7Deg2
elif act_str == 'swish_rg4_deg4':
    activation = activations.SwishRg4Deg4
elif act_str == 'swish_rg5_deg4':
    activation = activations.SwishRg5Deg4
elif act_str == 'swish_rg6_deg4':
    activation = activations.SwishRg6Deg4
elif act_str == 'swish_rg7_deg4':
    activation = activations.SwishRg7Deg4
# Mish approx.
elif act_str == 'mish_rg4_deg2':
    activation = activations.MishRg4Deg2
elif act_str == 'mish_rg5_deg2':
    activation = activations.MishRg5Deg2
elif act_str == 'mish_rg6_deg2':
    activation = activations.MishRg6Deg2
elif act_str == 'mish_rg7_deg2':
    activation = activations.MishRg7Deg2
elif act_str == 'mish_rg4_deg4':
    activation = activations.MishRg4Deg4
elif act_str == 'mish_rg5_deg4':
    activation = activations.MishRg5Deg4
elif act_str == 'mish_rg6_deg4':
    activation = activations.MishRg6Deg4
elif act_str == 'mish_rg7_deg4':
    activation = activations.MishRg7Deg4


# %%
now = datetime.datetime.now()
md_str = now.strftime('%m%d')
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if use_ghost_module:
    BASE_FILE_NAME = f"{os.path.splitext(os.path.basename(__file__))[0]}_ghost-{act_str}"
else:
    BASE_FILE_NAME = f"{os.path.splitext(os.path.basename(__file__))[0]}-{act_str}"
if bn_enabled:
    BASE_FILE_NAME += "-BN"
if do_enabled:
    BASE_FILE_NAME += "-DO"
if gap_enabled:
    BASE_FILE_NAME += "-GAP"
SAVE_MODEL_DIR_NAME = "saved_models"
PRUNE_MODEL_DIR_NAME = "pruned_models"
BEST_MODEL_STATE_DICT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-{md_str}-best.pt"
PRUNE_MODEL_STATE_DICT_BASE = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-prune_conv2_0.4-round"


# %%
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()

        conv1_list = []
        if use_ghost_module:
            conv1_list.append(GhostModule(in_channels=1, out_channels=5, kernel_size=5, stride=2))
        else:
            conv1_list.append(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2))
        if bn_enabled:
            conv1_list.append(nn.BatchNorm2d(5))
        conv1_list.append(activation())

        conv2_list = []
        if use_ghost_module:
            conv2_list.append(GhostModule(in_channels=5, out_channels=50, kernel_size=5, stride=2))
        else:
            conv2_list.append(nn.Conv2d(in_channels=5, out_channels=50, kernel_size=5, stride=2))
        if bn_enabled:
            conv2_list.append(nn.BatchNorm2d(50))
        conv2_list.append(activation())

        self.conv1 = nn.Sequential(*conv1_list)
        self.conv2 = nn.Sequential(*conv2_list)
        if do_enabled:
            self.dropout = nn.Dropout2d(0.25)
        if gap_enabled:
            self.gap = nn.AvgPool2d(kernel_size=4)
            self.fc = nn.Linear(50, 10)
        else:
            self.fc = nn.Linear(50 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        if do_enabled:
            x = self.dropout(x)
        if gap_enabled:
            x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_one_epoch(epoch, model, train_loader, val_loader, loss_func, optimizer, scheduler):
    model.train()

    total_step = len(train_loader)
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Clear the gradients of all parameters
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        batch_out = model(inputs)
        loss = loss_func(batch_out, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(batch_out, dim=-1)
        running_loss += loss.cpu().item()
        running_corrects += torch.sum(preds == labels.data).cpu().item()
        train_acc(preds.cpu(), labels.cpu())

        # Print statistics (per 100 iterations and end of epoch)
        if (i + 1) % 100 == 0 or (i + 1) == total_step:
            print(f"Step [{i+1:3d}/{total_step:3d}] -> Loss: {loss.item():.4f}")

    else:
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                val_out = model(val_inputs)
                val_loss = loss_func(val_out, val_labels)

                val_preds = torch.argmax(val_out, dim=-1)
                val_running_loss += val_loss.cpu().item()
                val_running_corrects += torch.sum(val_preds == val_labels.data).cpu().item()
                test_acc(val_preds.cpu(), val_labels.cpu())

        epoch_loss = running_loss / len(train_loader)  # loss per epoch
        epoch_acc = running_corrects / len(train_loader.dataset)  # accuracy per epoch
        running_loss_history.append(epoch_loss)
        running_acc_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_running_corrects / len(val_loader.dataset)
        val_running_loss_history.append(val_epoch_loss)
        val_running_acc_history.append(val_epoch_acc)

        print(f'training loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')
        print(f'validation loss: {val_epoch_loss:.4f}, validation acc: {val_epoch_acc:.4f}')
        print(f'TrainAcc: {train_acc.compute()}, TestAcc: {test_acc.compute()}')
        global max_test_acc
        if max_test_acc < test_acc.compute():
            print(f'Test acc improved from {max_test_acc} to {test_acc.compute()}')
            if args.mode == 'prune':
                # torch.save(model.state_dict(), f"{PRUNE_MODEL_STATE_DICT_BASE}{args.round}.pt")
                torch.save(model, f"{PRUNE_MODEL_STATE_DICT_BASE}{args.round}.pt")
            else:
                torch.save(model.state_dict(), BEST_MODEL_STATE_DICT_PATH)
            print('Model saved.')
            max_test_acc = test_acc.compute()
        sw.add_scalar('Train Accuracy', train_acc.compute(), epoch+1)
        sw.add_scalar('Test Accuracy', test_acc.compute(), epoch+1)
        scheduler.step(val_epoch_loss)


def test(model, data_loader):
    model.to(DEVICE)
    model.eval()  # Inference mode

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.max(outputs, dim=1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * float(correct / total)
    print(f"Accuracy: {acc:.2f}")
    return acc


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, INPUT_C, INPUT_H, INPUT_W))

    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    def prune_linear(linear, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(linear.weight, amount=amount)
        plan = DG.get_pruning_plan(linear, tp.prune_linear, pruning_index)
        plan.exec()

    if gap_enabled:
        linear_layer_prune_probs = [0.0, 0.4, 0.0]
    else:
        linear_layer_prune_probs = [0.0, 0.4, 0.3]
    idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune_conv(m, linear_layer_prune_probs[idx])
            idx += 1
        # elif isinstance(m, nn.Linear):
        #     prune_linear(m, linear_layer_prune_probs[idx])
        #     idx += 1

    return model


def main():
    print(f"Device: {DEVICE}\n")

    # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomRotation(10),
    #                                       transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=10, scale=(0.8, 1.2)),
    #                                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize(mean=(0.5,), std=(0.5,))])
    # train_transform = transforms.Compose([rand_augment_transform(config_str='rand-m4-n2-mstd0.5', hparams={'translate_const': 117}),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize(mean=(0.5,), std=(0.5,))])
    #                                     transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    train_transform = transforms.Compose([transforms.RandomRotation(15),
                                          transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    #                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    # Download MNIST dataset
    train_data = datasets.MNIST(root="./data", train=True, transform=train_transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    print("<Train data>")
    print(train_data)
    print()
    print(f"Train data images: {train_data.data.shape}")
    print(f"Train data labels: {len(train_data.targets)}\n")
    print("<Test data>")
    print(test_data)
    print()
    print(f"Test data images: {test_data.data.shape}")
    print(f"Test data labels: {len(test_data.targets)}\n")

    # Define data loaders
    loaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    model = MnistCNN()
    summary(model, input_size=(1, INPUT_C, INPUT_H, INPUT_W))
    print()

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if args.mode == 'train':
        # Train model
        model = model.to(DEVICE)
        for epoch in range(EPOCHS):
            print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
            train_acc.reset(), test_acc.reset()
            train_one_epoch(epoch, model, loaders["train"], loaders["test"], loss_func, optimizer, scheduler)
            print()

        print(f"Finished training! (Best accuracy: {max_test_acc})")
        print()

        # Test model
        best_model = MnistCNN()
        best_model.load_state_dict(torch.load(BEST_MODEL_STATE_DICT_PATH, map_location=torch.device(DEVICE)))
        best_model = best_model.to(DEVICE)
        start = time.perf_counter()
        test_accuracy = test(best_model, loaders["test"])
        end = time.perf_counter()
        print(f"Test time using model state dict: {end - start} [sec]\n")

        now = datetime.datetime.now()
        md_hm_str = now.strftime('%m%d_%H%M')
        test_acc_str = f'{test_accuracy:.2f}'
        common_name = f'{BASE_FILE_NAME}-{test_acc_str}_{EPOCHS}epoch-{md_hm_str}'
        MODEL_STRUCTURE_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-structure.json"
        MODEL_PARAMS_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{common_name}-params.h5"
        HISTORY_GRAPH_PATH = f"{CUR_DIR}/graphs/{common_name}_history.png"

        # Save model structure in JSON and params in hdf5
        save_structure_as_json(model, MODEL_STRUCTURE_PATH)
        best_model = best_model.to('cpu')
        save_params_as_hdf5(best_model, MODEL_PARAMS_PATH)

        # Plot history graph (accuracy & loss) and save
        plot_and_save_training_history(running_loss_history,
                                       val_running_loss_history,
                                       running_acc_history,
                                       val_running_acc_history,
                                       HISTORY_GRAPH_PATH)
    elif args.mode == 'prune':
        previous_ckpt = f"{PRUNE_MODEL_STATE_DICT_BASE}{args.round - 1}.pt"
        print(f"Pruning round {args.round}, load model from {previous_ckpt}")
        if args.round == 1:
            model = MnistCNN()
            model.load_state_dict(torch.load(previous_ckpt))
        else:
            model = torch.load(previous_ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        # print(f"Number of Parameters: {params/1e6:.4f}M (Before pruning)")
        print(f"Number of Parameters: {params} (Before pruning)")
        test(model, loaders["test"])
        model.to('cpu')
        prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print(f"Number of Parameters: {params} (After pruning)")
        test(model, loaders["test"])
        print("Start training...")
        model = model.to(DEVICE)
        for epoch in range(EPOCHS):
            print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
            train_acc.reset(), test_acc.reset()
            train_one_epoch(epoch, model, loaders["train"], loaders["test"], loss_func, optimizer, scheduler)
            print()

        print(f"Finished training! (Best accuracy: {max_test_acc})")
        print()
        pruned_model_structure_path = f"{PRUNE_MODEL_STATE_DICT_BASE}{args.round}-{max_test_acc*100:.2f}-structure.json"
        print(f"Save model struecture in {pruned_model_structure_path}")
        save_structure_as_json(model, pruned_model_structure_path)
        pruned_model_params_path = f"{PRUNE_MODEL_STATE_DICT_BASE}{args.round}-{max_test_acc*100:.2f}-params.h5"
        print(f"Save model parameters in {pruned_model_params_path}")
        model = model.to('cpu')
        save_params_as_hdf5(model, pruned_model_params_path)


if __name__ == '__main__':
    main()
