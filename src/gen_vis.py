import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def plt_roc(dim, y_pred, y_true, PathOutput):
    matplotlib.rcParams.update({'font.size': 18})
    n_classes=dim
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    file=PathOutput+"roc_curve.png"
    plt.savefig(file)
    plt.show()
    plt.close()
    print("AUC_Score=",roc_auc[0])

def plt_epoch(dim, y_pred, y_true, PathOutput):
    logfile=PathOutput+'allnode_PIN.log'
    df_log = pd.read_csv(logfile)
    x=df_log['epoch'].values
    y1=df_log['loss'].values
    y2=df_log['val_loss'].values
    
    fig = plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    ax = fig.add_subplot(111)
    ax.plot(x, y1, label="Test")
    ax.plot(x, y2, label="Validation")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #(handles, labels, loc='right', bbox_to_anchor=(0.5,-0.1))
    ax.grid('on')
    filename=PathOutput+"cross_entropy.png"
    fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()
    
    x=df_log['epoch'].values
    y1=df_log['acc'].values
    y2=df_log['val_acc'].values
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    ax = fig.add_subplot(111)
    ax.plot(x, y1, label="Test")
    ax.plot(x, y2, label="Validation")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #(handles, labels, loc='right', bbox_to_anchor=(0.5,-0.1))
    ax.grid('on')
    filename=PathOutput+"accuracy.png"
    fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

def cls_rep(y_pred, y_true, target_names):
    print(classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), target_names=target_names,digits=4))
    print(confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)))
    
def cls_rep_nm(y_pred, y_true, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names,digits=4))
    print(confusion_matrix(y_true, y_pred))
