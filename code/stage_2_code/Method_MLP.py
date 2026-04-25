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

    def evaluate_dataset(self, X_tensor, y_true, loss_function, evaluator_name):
        accuracy_evaluator = Evaluate_Accuracy(evaluator_name, '')

        super().train(False)
        with torch.no_grad():
            y_pred = self.forward(X_tensor.to(self.device))
            loss = loss_function(y_pred, y_true.to(self.device)).item()
            pred_labels = y_pred.max(1)[1].cpu()

        accuracy_evaluator.data = {
            'true_y': y_true,
            'pred_y': pred_labels
        }
        acc = accuracy_evaluator.evaluate()
        return loss, acc

    def train(self, X, y, test_X, test_y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        train_X_tensor = torch.FloatTensor(np.array(X))
        train_y_true = torch.LongTensor(np.array(y))
        test_X_tensor = torch.FloatTensor(np.array(test_X))
        test_y_true = torch.LongTensor(np.array(test_y))

        train_dataset = TensorDataset(train_X_tensor, train_y_true)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        epoch_list = []
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

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

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_eval_loss, train_acc = self.evaluate_dataset(
                train_X_tensor, train_y_true, loss_function, 'training evaluator'
            )
            test_loss, test_acc = self.evaluate_dataset(
                test_X_tensor, test_y_true, loss_function, 'testing evaluator'
            )

            epoch_list.append(epoch)
            train_loss_list.append(avg_epoch_loss if epoch == 0 else train_eval_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)


            print('Epoch:', epoch,
                    'Train Accuracy:', train_acc,
                    'Train Loss:', train_loss_list[-1],
                    'Test Accuracy:', test_acc,
                    'Test Loss:', test_loss)

        self.save_learning_curves(epoch_list, train_loss_list, train_acc_list,
                                  test_loss_list, test_acc_list)
        print('Final Train Loss:', train_loss_list[-1])
        print('Final Test Loss:', test_loss_list[-1])

    def save_learning_curves(self, epochs, train_losses, train_accs, test_losses, test_accs):
        save_folder = '../../result/stage_2_result/'
        os.makedirs(save_folder, exist_ok=True)

        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Test Loss Curve')
        plt.legend()
        plt.savefig(save_folder + 'training_loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accs, label='Train Accuracy')
        plt.plot(epochs, test_accs, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train vs Test Accuracy Curve')
        plt.legend()
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
        self.train(
            self.data['train']['X'], self.data['train']['y'],
            self.data['test']['X'], self.data['test']['y']
        )
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
