# d-vector
import keras.backend as K
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout,Activation,merge,ZeroPadding2D
from keras.layers import Dense,Lambda,Add,GlobalAveragePooling2D,ZeroPadding2D,Multiply,GlobalMaxPool2D,Dropout
from keras.regularizers import l2
from keras import Model
from keras.layers.core import Permute
from keras import regularizers
from keras.layers import Conv1D,MaxPool1D,LSTM
from keras import initializers
from keras.layers import GlobalMaxPool1D,Permute
from keras.layers import GRU,TimeDistributed,Flatten, LeakyReLU,ELU

import  keras.backend as K

class D_Vector():
    def d_vector(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        x = Flatten(name='flatten')(x_in)
        
        # hidden1
        x = Dense(4096,name='fc1')(x)
        
        x = BatchNormalization(name='bn_fc1')(x)  
        
        x = Activation('relu',name='fc1_relu')(x)
        
        x = Dropout(0.5,name='drop_1')(x)
        
        # # # hidden2
        # x = Dense(1024,name='fc2')(x)
        
        # x = BatchNormalization(name='bn_fc2')(x)  
        
        # x = Activation('relu',name='fc2_relu')(x)
        
        # # x = Dropout(0.5,name='drop_2')(x)
        
        # # hidden3
        x = Dense(1024,name='fc3')(x)
        
        x = BatchNormalization(name='bn_fc3')(x)  
        
        x = Activation('relu',name='fc3_relu')(x)
        
        x = Dropout(0.5,name='drop_3')(x)

        
        # # # hidden4
        x = Dense(512,name='fc4')(x)
        
        x = BatchNormalization(name='bn_fc4')(x)  
        
        x = Activation('relu',name='fc4_relu')(x)
        
        x = Dropout(0.5,name='drop_4')(x)
      
        model = Model(inputs=[x_in],outputs=[x],name='d-vector')
        
        # model.summary()
        return model
    
if __name__ == "__main__":
    model = D_vector()
    input_shape = (299,40,1)
    model = model.d_vector(input_shape)
    print(model.summary())