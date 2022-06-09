"""
cite:

https://autokeras.com/tutorial/overview/

@inproceedings{jin2019auto,
  title={Auto-Keras: An Efficient Neural Architecture Search System},
  author={Jin, Haifeng and Song, Qingquan and Hu, Xia},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1946--1956},
  year={2019},
  organization={ACM}
}

"""
import datetime
import os
import librosa
import numpy as np
import autokeras as ak
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from alf.alf.ALFpaka.core.helper import helper, stft


# create time based filename
tc = str(datetime.datetime.now())
tc = tc.replace(":", "-")
tc = tc.replace(" ", "-")
tc = tc.replace(".", "-")
path = os.getcwd()
outfile = os.path.join(path, (tc + ".txt"))


def printout(content):
    print(content)
    with open(outfile, 'a') as logfile:
        print(content, file=logfile)


def get_magnitude_spec(x, sample_rate, fftsize, winsize, hopsize, highest_bin, lowest_bin, log_spec, mel_bands,
                       avg_frames):
    X, phase = stft.STFT(hopsize=hopsize, winsize=winsize, fftsize=fftsize).getMagnAndPhase(x)
    if highest_bin is None:
        highest_bin = X.shape[1]
    X = X[:, lowest_bin:highest_bin]

    if mel_bands > 0:
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=fftsize,
            n_mels=mel_bands,
            fmin=0,
            fmax=int(sample_rate/2.0),
            htk=False).T
        X = np.dot(X, mel_basis)

    if log_spec is True:
        X = np.log(1+X)

    if avg_frames > 0:
        X = helper.moving_average_filtering_only_backwards(X, avg_frames)
    return X[0]


def collect_rooms(rootPath, rList, sr, mono=True, specFeat=128, mixSetupPos = True, mixArray = True,
                  mixSourcePos = True):
    printout(str(datetime.datetime.now()) + ": Starting Dataset creation ...")

    setupPosList = ['_E', '_M', '_W']
    if not mixSetupPos:
        setupPosList = list([setupPosList[0]])
    arrayList = ['_DH', '_MTB', '_SDM']
    if not mixArray:
        arrayList = list([arrayList[0]])
    sourcePosList = ['_C', '_L', '_R', '_LS', '_RS']
    if not mixSourcePos:
        sourcePosList = list([sourcePosList[0]])

    angleList = ['_000', '_005', '_010', '_015', '_020', '_025', '_030', '_035', '_040', '_045', '_050', '_055',
                 '_060', '_065', '_070', '_075', '_080', '_085', '_090', '_095', '_100', '_105', '_110', '_115',
                 '_120', '_125', '_130', '_135', '_140', '_145', '_150', '_155', '_160', '_165', '_170', '_175',
                 '_180', '_185', '_190', '_195', '_200', '_205', '_210', '_215', '_220', '_225', '_230', '_235',
                 '_240', '_245', '_250', '_255', '_260', '_265', '_270', '_275', '_280', '_285', '_290', '_295',
                 '_300', '_305', '_310', '_315', '_320', '_325', '_330', '_335', '_340', '_345', '_350', '_355']
    suffix = '.wav'

    locList = []
    gtList1hot = []
    gtList = []
    for room in rList:
        # create GT vector
        onehot = np.zeros((1, len(rList)), dtype=np.int8)
        hotPos = rList.index(room)
        onehot[0][hotPos] = 1
        gtVec = onehot[0]
        for setupPos in setupPosList:
            for array in arrayList:
                for sourcePos in sourcePosList:
                    for angle in angleList:
                        if array == '_SDM':
                            filename = 'BRIR' + array + '-KEMAR' + room + setupPos + sourcePos + angle + suffix
                            location = os.path.join(rootPath, room[1:], setupPos[1:], 'BRIRs', array[1:], 'KEMAR',
                                                    (room[1:] + sourcePos + setupPos), 'Quantized50DOA', filename)
                        else:
                            filename = 'BRIR' + array + room + setupPos + sourcePos + angle + suffix
                            location = os.path.join(rootPath, room[1:], setupPos[1:], 'BRIRs', array[1:], filename)
                        #assert(os.path.exists(location), "Could not find file: " + str(location))
                        locList.append(location)
                        gtList.append(hotPos)
                        gtList1hot.append(gtVec)

    # load audio
    printout(str(datetime.datetime.now()) + ": Loading BRIR files from disk ...")
    X = []
    minLength = 480000
    for locname in locList:
        x, sr = librosa.load(locname, mono=False, sr=sr)
        if mono:
            x = x[0]
        x = x/np.max(np.abs(x))
        X.append(x)
        xlength = len(x)
        if xlength < minLength:
            minLength = xlength

    # correct length
    for n in range(len(X)):
        if specFeat > 0:
            X[n] = get_magnitude_spec(X[n][0:minLength], sr, minLength, minLength, minLength, None, None, True,
                                      specFeat, 0)
        else:
            X[n] = X[n][0:minLength]

    if len(gtVec) < 3:
        y_out = gtList
    else:
        y_out = gtList1hot

    printout(str(datetime.datetime.now()) + ": Dataset Extraction finished.")
    return [np.asarray(X), np.asarray(y_out)]


if __name__ == "__main__":

    # init
    fs = 48000
    mel_bands = 512
    mixSetupPos = False
    mixArray = False
    mixSourcePos = False
    test_size = 0.2
    max_trials = 3
    num_epochs = 10

    # load data
    path2data = []
    roomlist = ['_H1539b', '_H1562'] #, '_H2505', '_HL', '_HU103', '_ML2-102']
    if os.name == 'nt':
        path2data = 'H:/workspace/TUI/room_similarity/data/Daten/'
    else:
        path2data = '/home/kehlcn/workspace/TUI/room_similarity/data/Daten/'
    X, Y = collect_rooms(path2data, roomlist, fs, True, mel_bands, mixSetupPos=mixSetupPos, mixArray=mixArray,
                         mixSourcePos=mixSourcePos)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    printout("\n\nSamplerate: " + str(fs))
    printout("Number of Mel Bands: " + str(mel_bands))
    printout("Is Setup position mixed: " + str(mixSetupPos))
    printout("Are Arrays mixed: " + str(mixArray))
    printout("Is Source position mixed: " + str(mixSourcePos))

    printout("\n\nShape of Training Data (X_train): " + str(X_train.shape))
    printout("Shape of Training Labels (X_train): " + str(y_train.shape))
    printout("Shape of Test Data (X_train): " + str(X_test.shape))
    printout("Shape of Test Labels (X_train): " + str(y_test.shape))
    printout("Data Split Ratio: " + str((1 - test_size) * 100) + "% train : " + str(test_size * 100) + "% test")

    # autokeras init
    search = ak.StructuredDataClassifier(
        column_names=None,
        column_types=None,
        num_classes=len(roomlist),
        multi_label=False,
        loss=None,
        metrics=None,
        project_name=tc,
        max_trials=max_trials,
        directory=None,
        objective="val_accuracy",
        tuner=None,
        overwrite=False,
        seed=None,
        max_model_size=None #, **kwargs
    )

    # train models
    search.fit(x=X_train,
               y=y_train,
               verbose=1,
               validation_split=0.2,
                epochs=num_epochs)

    # evaluate the trained models
    loss, acc = search.evaluate(X_test, y_test, verbose=1) #classification
    # mae, _ = search.evaluate(X_test, y_test, verbose=0) #regression
    y_predictions = search.predict(X_test, verbose=1)
    if len(y_test.shape) < 2:
        y_predictions = np.max(y_predictions, axis=1)
    # save/write out
    model = search.export_model()
    modelname = tc + '_best_model.h5'
    model.save(modelname)
    model.summary(print_fn=printout)

    # statistics print out
    report = classification_report(y_test, y_predictions, target_names=roomlist)
    acc_test = accuracy_score(y_test, y_predictions)
    f1_test = f1_score(y_test, y_predictions, average='macro')
    if len(y_test.shape) > 1:
        cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predictions, axis=1))
    else:
        cm = confusion_matrix(y_test, y_predictions)
    cmsum = 1/cm.sum(axis=1)
    cm_norm = np.multiply(cm, cmsum[:, None])
    printout(report)
    printout(cm_norm)
    printout("Test accuracy = " + str(acc_test))
    printout("Test F1-Score = " + str(f1_test))
    printout("\n\n" + str(datetime.datetime.now()))
    printout("EOF")

    # plot conf matrix
    cmd = ConfusionMatrixDisplay(cm_norm, display_labels=roomlist)
    cmd.plot(cmap='binary', values_format='.2%')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_'+str(tc)+'.png')

    # plot roc curve
    if len(y_test.shape) < 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_predictions)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
        display.plot()
        plt.savefig('ROC_curve_' + str(tc) + '.png')

