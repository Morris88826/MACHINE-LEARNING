import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def verify_pred_svm(data1, data2):
    
    TP = np.sum((data1 == 1) & (data2 == 1))
    TN = np.sum((data1 == 0) & (data2 == 0))
    FP = np.sum((data1 == 0) & (data2 == 1))
    FN = np.sum((data1 == 1) & (data2 == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)


    print("Accuracy: {:.3f}".format(accuracy))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1 Score: {:.3f}".format(f1_score))

    
    # plot confusion matrix
    cm = confusion_matrix(data1, data2)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify the data')
    parser.add_argument('--input_1', type=str, help='Input file 1')
    parser.add_argument('--input_2', type=str, help='Input file 2')
    args = parser.parse_args()

    df1 = pd.read_csv(args.input_1)
    df2 = pd.read_csv(args.input_2)

    assert df1.shape[0] == df2.shape[0], "The number of rows is different"

    verify_pred_svm(df1["pred_svm"], df2["pred_svm"])