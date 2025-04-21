import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(true_labels, predicted_labels, average='weighted'):
    accuracy = accuracy_score(true_labels, predicted_labels)
    # precision = precision_score(true_labels, predicted_labels, average=average)
    # recall = recall_score(true_labels, predicted_labels, average=average)
    # f1 = f1_score(true_labels, predicted_labels, average=average)
    # error_rate = 1 - accuracy
    return accuracy
    # return accuracy, precision, recall, f1, error_rate

    
def plot_loss_over_epochs(epochs,status,value, loss_or_acc, data_name):
    
    plt.plot(epochs, value)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(data_name + '_' + status +'_' + loss_or_acc)
    plt.savefig("./{}_plot_{}_{}.png".format(data_name, status, loss_or_acc))
    plt.clf()
