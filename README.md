# Deopen
A hybrid deep convolutional neural network for predicting chromatin accessibility

# Requirements
- h5py
- hickle
- Scikit-learn=0.18.2
- Theano=0.8.0
- Lasagne=0.2.dev1
- nolearn=0.6.0

# Training

 
```
# Run Deopen classification model
THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_classification.py -in <input_file.hkl> -out <outputfile>

# Run Deopen regression model
THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_regression.py -in <inputfile.hkl> -reads <readsfile.hkl> -out <outputfile>

#TBA
```


# License
This project is licensed under the MIT License - see the LICENSE.md file for details
