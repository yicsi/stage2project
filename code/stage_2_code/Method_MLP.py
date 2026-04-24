'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 10)

    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        y_pred = self.fc_layer_3(h)
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        epoch_list = []
        loss_list = []
        acc_list = []

        for epoch in range(self.max_epoch):
            X_tensor = torch.FloatTensor(np.array(X))
            y_true = torch.LongTensor(np.array(y))

            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            accuracy_evaluator.data = {
                'true_y': y_true,
                'pred_y': y_pred.max(1)[1]
            }
            train_acc = accuracy_evaluator.evaluate()

            epoch_list.append(epoch)
            loss_list.append(train_loss.item())
            acc_list.append(train_acc)

            if epoch % 10 == 0:
                print('Epoch:', epoch,
                      'Accuracy:', train_acc,
                      'Loss:', train_loss.item())

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
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}