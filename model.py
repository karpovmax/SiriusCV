#import cPickle as pickle
import pickle
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Concatenate, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.python.keras.optimizers import Adam, Adamax
import numpy as np
from os.path import isfile

INPUTSIZE = (256, 512, 3)


def blockDown(inp,
              filters,  # conv parameters
              kernelSize=(3, 3),
              stridesConv=(1, 1),
              paddingConv='same',
              dataFormatConv=None,
              dilationRate=(1, 1),
              activation='relu',
              useBias=True,
              kernelInitializer='he_normal',
              biasInitializer='zeros',
              kernelRegularizer=None,
              biasRegularizer=None,
              activityRegularizer=None,
              kernelConstraint=None,
              biasConstraint=None,
              poolSize=(2, 2),  # pool parameters
              stridesPool=None,
              paddingPool='valid',
              dataFormatPool=None,
              dropRate=None,    # drop parameters
              noiseShape=None,
              seed=None):
    
    conv = Conv2D(filters=filters,
                         kernel_size=kernelSize,
                         strides=stridesConv,
                         padding=padding,
                         data_format=dataFormatConv,
                         dilation_rate=dilationRate,
                         activation=activation,
                         use_bias=useBias,
                         kernel_initializer=kernelInitializer,
                         bias_initializer=biasInitializer,
                         kernel_regularizer=kernelRegularizer,
                         bias_regularizer=biasRegularizer,
                         activity_regularizer=activityRegularizer,
                         kernel_constraint=kernelConstraint,
                         bias_constraint=biasConstraint)(inp)
    conv = Conv2D(filters=filters,
                         kernel_size=kernelSize,
                         strides=stridesConv,
                         padding=padding,
                         data_format=dataFormatConv,
                         dilation_rate=dilationRate,
                         activation=activation,
                         use_bias=useBias,
                         kernel_initializer=kernelInitializer,
                         bias_initializer=biasInitializer,
                         kernel_regularizer=kernelRegularizer,
                         bias_regularizer=biasRegularizer,
                         activity_regularizer=activityRegularizer,
                         kernel_constraint=kernelConstraint,
                         bias_constraint=biasConstraint)(conv)
    if dropRate != None:
        conv = Dropout(rate=dropRate,
                              noise_shape=noiseShape,
                              seed=seed)(conv)
    pool = MaxPool2D(pool_size=poolSize,
                            strides=stridesPool,
                            padding=paddingPool,
                            data_format=dataFormatPool)(conv)
    
    return pool, conv   # pool for next step and conv is the layer which we have to conctenate at next step


def blockUp(inp,
            toConc,
            filters,    # conv parameters
            kernelSize=(3, 3),
            stridesConv=(1, 1),
            paddingConv='same',
            dataFormatConv=None,
            dilationRate=(1, 1),
            activation='relu',
            useBias=True,
            kernelInitializer='he_normal',
            biasInitializer='zeros',
            kernelRegularizer=None,
            biasRegularizer=None,
            activityRegularizer=None,
            kernelConstraint=None,
            biasConstraint=None,
            upSize=(2, 2),  # up parameters
            dataFormatUp=None):

    up = UpSampling(size=upSize,
                           data_format=dataFormat)(inp)
    conc = Concatenate()([up, toConc])
    conv = Conv2D(filters=filters,
                         kernel_size=kernelSize,
                         strides=stridesConv,
                         padding=padding,
                         data_format=dataFormatConv,
                         dilation_rate=dilationRate,
                         activation=activation,
                         use_bias=useBias,
                         kernel_initializer=kernelInitializer,
                         bias_initializer=biasInitializer,
                         kernel_regularizer=kernelRegularizer,
                         bias_regularizer=biasRegularizer,
                         activity_regularizer=activityRegularizer,
                         kernel_constraint=kernelConstraint,
                         bias_constraint=biasConstraint)(conc)

    return conv


def unet(inputSize=INPUTSIZE,
         batchSize=None,
         name=None,
         dtype=None,
         sparse=False,
         tensor=None,
         pretrainedWeights=None):
    inp = Input(shape=inputSize,
                batch_size=batchSize,
                name=name,
                dtype=dtype,
                sparse=sparse,
                tensor=tensor)
    down1, toConc1 = blockDown(inp=inp,
                               filters=8)
    down2, toConc2 = blockDown(inp=down1,
                               filters=16)
    down3, toConc3 = blockDown(inp=down2,
                               filters=32)
    down4, toConc4 = blockDown(inp=down3,
                               filters=64,
                               dropRate=0.5)
    down5, toConc5 = blockDown(inp=down4,
                               filters=128,
                               dropRate=0.5)
    up6 = blockUp(inp=down5,
                  toConc=toConc5,
                  filters=64)
    up7 = blockUp(inp=up6,
                  toConc=toConc4,
                  filters=32)
    up8 = blockUp(inp=up7,
                  toConc=toConc3,
                  filters=16)
    up9 = blockUp(inp=up8,
                  toConc=toConc2,
                  filters=8)
    finish = Conv2D(filters=10,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu',
                    kernel_initializer='he_normal')(up9)

    model = Model(inp, finish)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    if pretrainedWeights != None:
        model.load_weights(pretrainedWeights)

    return model


class model:

    def __init__(self,      #we are required to determine all these variables to pass them in tf.Model
                 
                 _proportionTrainVal=0.8,
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
                 
                 _validationSteps=20):
        
        self.proportionTrainVal = _proportionTrainVal
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
        self.validationSteps = _validationSteps

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

        datasetTrain = get_dataset(featuresTrain,
                                   labelsTrain)  # features and labels are arrays or lists? and other parameters
        datasetVal = get_dataset(featuresVal,
                                 labelsVal)

        self.model = unet()
        self.model.fit(x=datasetTrain,
                       y=None,  # because x is dataset which contains y
                       batch_size=self.batchSize,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       callbacks=self.callbacks,
                       validation_split=self.validationSplit,
                       validation_data=datasetVal,
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

