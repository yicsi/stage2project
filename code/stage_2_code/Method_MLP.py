'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 256

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc_layer_1 = nn.Linear(784, 512)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(512, 256)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(256, 10)

        self.to(self.device)

    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        y_pred = self.fc_layer_3(h)
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_true = torch.LongTensor(np.array(y))
        train_dataset = TensorDataset(X_tensor, y_true)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        epoch_list = []
        loss_list = []
        acc_list = []

        for epoch in range(self.max_epoch):
            super().train(True)
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                y_pred = self.forward(batch_X)
                train_loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_X.size(0)

            super().train(False)
            with torch.no_grad():
                full_X = X_tensor.to(self.device)
                full_pred = self.forward(full_X)
                pred_labels = full_pred.max(1)[1].cpu()

            accuracy_evaluator.data = {
                'true_y': y_true,
                'pred_y': pred_labels
            }
            train_acc = accuracy_evaluator.evaluate()
            avg_epoch_loss = epoch_loss / len(train_dataset)

            epoch_list.append(epoch)
            loss_list.append(avg_epoch_loss)
            acc_list.append(train_acc)

            if epoch % 2 == 0:
                print('Epoch:', epoch,
                      'Accuracy:', train_acc,
                      'Loss:', avg_epoch_loss)

        self.save_learning_curves(epoch_list, loss_list, acc_list)

    def save_learning_curves(self, epochs, losses, accs):
        save_folder = '../../result/stage_2_result/'
        os.makedirs(save_folder, exist_ok=True)

        plt.figure()
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig(save_folder + 'training_loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(epochs, accs)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.savefig(save_folder + 'training_accuracy_curve.png')
        plt.close()

    def test(self, X):
        super().train(False)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
            y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1].cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
