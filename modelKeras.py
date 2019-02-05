#import cPickle as pickle
import pickle
from keras.models import Model
import numpy as np
from os.path import isfile


def unet():



class model:
    
    def __init__(self,      #we are required to determine all these variables to pass them in tf.Model
                 _batchSize=None,
                 _epochs=8,
                 _verbose=1,
                 _callbacks=None,
                 _validationSplit=0.0,
                 _shuffle=True,
                 _classWeight=None,
                 _sampleWeight=None,
                 _initialEpoch=0,
                 _stepsPerEpoch=1500,
                 _validationSteps=20,
                 _proportionTrainVal=0.8): 
        self.batchSize = _batchSize
        self.epochs = _numberOfEpochs
        self.verbose = _verbose
        self.callbacks = _callbacks
        self.validationSplit = _validationSplit
        self.shuffle = _shuffle
        self.classWeight = _classWeight
        self.sampleWeight = _sampleWeight
        self.initialEpoch = _initialEpoch
        self.stepsPerEpoch = _stepsPerEpoch
        self.validationSteps = _validationStep

        self.proportionTrainVal = _proportionTrainVal
        self.model = Model()
        self.modelCheckpoint = 0    # we have to decide about modelCheckpoint type. and we really want to use it if we have callbacks?
        if (_callbacks != None):
            self.modelCheckpoint = self.callbacks[-1]   # i believe
    
    def fit(self, X, y):
        borderTrainVal = y.shape[0] * self.proportionTrainVal
        featuresTrain = np.array(X[:borderTrainVal])
        labelsTrain = np.array(y[:borderTrainVal])
        featuresVal = np.array(X[borderTrainVal:])
        labelsVal = np.array(y[borderTrainVal:])
        
        self.model.fit(x=featuresTrain,
                       y=labelsTrain,
                       batch_size=self.batchSize,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       callbacks=self.callbacks,
                       validation_split=self.validationSplit,
                       validation_data=(featuresVal, labelsVal),
                       shuffle=self.shuffle,
                       class_weight=self.classWeight,
                       sample_weight=self.sampleWeight,
                       initial_epoch=self.initialEpoch,
                       steps_per_epoch=self.stepsPerEpoch,
                       validation_steps=self.validationSteps)

    def predict(self, X):
        pred = self.mod.predict(X)
        return pred
    
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self



#if __name__ == "main":     dont forget to add it
