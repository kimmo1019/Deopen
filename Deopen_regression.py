'''
This script is used for running Deopen regression model.
Usage:
    THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_regression.py -in inputfile.hkl -reads readsfile.hkl -out outputfile
'''
import hickle as hkl
import argparse
from sklearn.cross_validation import ShuffleSplit
from scipy import stats
import numpy as np
import hickle as hkl
import theano
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import SliceLayer
from lasagne.layers import FlattenLayer
from lasagne.layers import ConcatLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print 'Using Lasagne.layers.dnn (faster)'
except ImportError:
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer
    print 'Using Lasagne.layers (slower)'
from lasagne.nonlinearities import softmax
from lasagne.objectives import squared_error
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from sklearn import metrics
floatX = theano.config.floatX



def data_split(inputfile,reads_feature):
    data = hkl.load(inputfile)
    reads_count = hkl.load(reads_feature)
    X = data['mat']
    X_kspec = data['kmer']
    y = np.mean(reads_count+1)
    rs = ShuffleSplit(len(y), n_iter=1,random_state = 1)
    X_kspec = X_kspec.reshape((X_kspec.shape[0],1024,4))
    X = np.concatenate((X,X_kspec), axis = 1)
    X = X[:,np.newaxis]
    X = X.transpose((0,1,3,2))
    for train_idx, test_idx in rs:
        X_train = X[train_idx,:]
        y_train = y[train_idx]
        X_test = X[test_idx,:]
        y_test = y[test_idx]
    X_train = X_train.astype('float32')
    y_train = y_train.astype('int32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('int32')
    return [X_train, y_train, X_test, y_test]

def model_train(X_train, y_train,learning_rate = 1e-4,epochs = 10):
    l = 1000
    layer1 = InputLayer(shape=(None, 1, 4, l+1024))
    layer2_1 = SliceLayer(layer1, indices=slice(0, l), axis = -1)
    layer2_2 = SliceLayer(layer1, indices=slice(l, None), axis = -1)
    layer2_3 = SliceLayer(layer2_2, indices = slice(0,4), axis = -2)
    layer2_f = FlattenLayer(layer2_3)
    layer3 = Conv2DLayer(layer2_1,num_filters = 64, filter_size = (4,7))
    layer4 = Conv2DLayer(layer3,num_filters = 64, filter_size = (1,7))
    layer5 = Conv2DLayer(layer4,num_filters = 64, filter_size = (1,7))
    layer6 = MaxPool2DLayer(layer5, pool_size = (1,6))
    layer7 = Conv2DLayer(layer6,num_filters = 64, filter_size = (1,7))
    layer8 = Conv2DLayer(layer7,num_filters = 64, filter_size = (1,7))
    layer9 = Conv2DLayer(layer8,num_filters = 64, filter_size = (1,7))
    layer10 = MaxPool2DLayer(layer9, pool_size = (1,6))
    layer11 = Conv2DLayer(layer10,num_filters = 64, filter_size = (1,7))
    layer12 = Conv2DLayer(layer11,num_filters = 64, filter_size = (1,7))
    layer13 = Conv2DLayer(layer12,num_filters = 64, filter_size = (1,7))
    layer14 = MaxPool2DLayer(layer13, pool_size = (1,6))
    layer14_d = DenseLayer(layer14, num_units= 64)
    layer3_2 = DenseLayer(layer2_f, num_units = 64)
    layer15 = ConcatLayer([layer14_d,layer3_2])
    #layer15 = ConcatLayer([layer10_d,])
    layer16 = DropoutLayer(layer15)
    layer17 = DenseLayer(layer16, num_units=32)
    network = DenseLayer(layer17, num_units= 2, nonlinearity=None)
    lr = theano.shared(np.float32(learning_rate))
    net = NeuralNet(
                network,
                max_epochs=epochs,
                update=adam,
                update_learning_rate=lr,
                regression = True,
                train_split=TrainSplit(eval_size=0.1),
                objective_loss_function = squared_error,
                #on_epoch_finished=[AdjustVariable(lr, target=1e-8, half_life=20)],
                verbose=4)
    net.fit(X_train, y_train)
    return net

def model_test(net, X_test, y_test):
    #net.load_params_from('/path/to/weights_file')
    y_pred = net.predict(X_test)
    print stats.linregress(y_test,y_pred[:,0])

def save_model(model,outputfile):
    net.save_params_to(open(outputfile,'w'))


if  __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Deopen regression model') 
    parser.add_argument('-in', dest='input', type=str, help='file of preprocessed data')
    parser.add_argument('-reads', dest='readsfile', type=str, help='file of reads features')
    parser.add_argument('-out', dest='output', type=str, help='output file')
    args = parser.parse_args()
    X_train, y_train, X_test, y_test = data_split(args.input,args.readsfile)
    model = model_train(X_train, y_train)
    #save_model(model, args.output)
    model_test(model, X_test, y_test)


