'''
Concrete Evaluate class for a specific evaluation metrics
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        if self.data.get('final', False):
            print('******** Final Evaluation Metrics ********')
            print('Accuracy:', acc)
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1-score:', f1)

        return acc
