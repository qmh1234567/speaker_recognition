# VggVox
import keras.backend as K
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout,Activation,merge,ZeroPadding2D
from keras.layers import Dense,Lambda,Add,GlobalAveragePooling2D,ZeroPadding2D,Multiply,GlobalMaxPool2D
from keras.regularizers import l2
from keras import Model
from keras.layers.core import Permute
from keras import regularizers
from keras.layers import Conv1D,MaxPool1D,LSTM
from keras import initializers
from keras.layers import GlobalMaxPool1D,Permute
from keras.layers import GRU,TimeDistributed,Flatten, LeakyReLU,ELU

import  keras.backend as K

class VggVox():
    
    def __init__(self):
      self.WEIGHT_DECAY =  0.00001 
      
    def basic_block(self,x,filters,strides,name):
         #block a
        x = Conv2D(filters,(3,3),strides=strides,padding='same',
            kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY),name=f'{name}_conva')(x)
        
        x = BatchNormalization(name=f'{name}_bna')(x)
        
        x = Activation('relu',name=f'{name}_relua')(x)
    
        #block b
        x = Conv2D(filters,(3,3),strides=(1,1),padding='same',
            kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY),name=f'{name}_convb')(x)
        
        x = BatchNormalization(name=f'{name}_bnb')(x) 
        
        # shortcut
        shortcut = Conv2D(filters,(1,1),strides=strides,padding='same',name=f'{name}_shcut')(x)
    
        shortcut = BatchNormalization(name=f'{name}_stbn')(x)
        
        x = Add(name=f'{name}_add')([x,shortcut])
        
        x = Activation('relu',name=f'{name}_relu')(x)
        
        return x
    
    def res_34(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        # 改了一下步长
        x = Conv2D(64,(7,7),strides=(1,1),padding='same',name='conv1',kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x_in)
        
        x = BatchNormalization(name="bn1")(x)
    
        x = Activation('relu')(x)
        
        x = MaxPool2D((3,3),strides=(2,2),padding='same',name='pool1')(x)
        
        for i in range(1,4):
            x = self.basic_block(x,64,(1,1),name=f'block{i}')
            
        for i in range(4,8):
            stride = (2,2) if i==4 else (1,1)
            x = self.basic_block(x,128,stride,name=f'block{i}')
    
        for i in range(8,14):
            stride = (2,2) if i==8 else (1,1)
            x = self.basic_block(x,256,stride,name=f'block{i}')

        for i in range(14,17):
            stride = (2,2) if i==14 else (1,1)
            x = self.basic_block(x,512,stride,name=f'block{i}')
        
        # avgpool
        x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)

        x = Dense(512,name='fc1')(x)
        
        # x = BatchNormalization(name='bn_fc1')(x)
        
        # x = Activation('relu',name=f'relu1')(x) 
        
        # # 归一化
        x = Lambda(lambda y:K.l2_normalize(y,axis=1),name='ln')(x)
        
        model = Model(inputs=[x_in],outputs=[x],name='res-34')
        
        # model.summary()
        return model

    def res_conv_block(self,x,outdim,strides,name):
        
        # block a
        x = Conv2D(outdim//4,(1,1),strides=strides,kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY),name=f'{name}_conva')(x)
        x = BatchNormalization(name=f'{name}_bna')(x)
        x = Activation('relu',name=f'{name}_relua')(x)
        
        # block b
        x = Conv2D(outdim//4,(3,3),padding='same',kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY),name=f'{name}_convb')(x)
        x = BatchNormalization(name=f'{name}_bnb')(x)
        x = Activation('relu',name=f'{name}_relub')(x)
        
        # block c
        x = Conv2D(outdim,(1,1),name=f'{name}_convc',kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x)
        x = BatchNormalization(name=f'{name}_bnc')(x)
        
        # shortcut
        shortcut = Conv2D(outdim,(1,1),strides=strides,name=f'{name}_shcut',kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x)
        shortcut = BatchNormalization(name=f'{name}_stbn')(x)
        
        x = Add(name=f'{name}_add')([x,shortcut])
        x = Activation('relu',name=f'{name}_relu')(x)
        return x
        
    
    # ResNet50
    def res_50(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        x = Conv2D(64,(7,7),strides=(2,2),padding='same',name='conv1',kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x_in)
        
        x = BatchNormalization(name="bn1")(x)
    
        x = Activation('relu')(x)
        
        x = MaxPool2D((3,3),strides=(2,2),padding='same',name='pool1')(x)
        
        
        for i in range(1,4):
            x = self.res_conv_block(x,256,(1,1),name=f'block{i}')
            
        for i in range(4,8):
            stride = (2,2) if i==4 else (1,1)
            x = self.res_conv_block(x,512,stride,name=f'block{i}')
    
        for i in range(8,14):
            stride = (2,2) if i==8 else (1,1)
            x = self.basic_block(x,1024,stride,name=f'block{i}')

        for i in range(14,17):
            stride = (2,2) if i==14 else (1,1)
            x = self.basic_block(x,2048,stride,name=f'block{i}')

        x = Conv2D(512,(x.shape[1],1),name='fc6')(x)
        
        x = BatchNormalization(name="bn_fc6")(x)
        
        x = Activation('relu',name='relu_fc6')(x)
        
        # avgpool
        x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)
        
        model = Model(inputs=[x_in],outputs=[x],name='res-50')
        
        return model



if __name__ == "__main__":
    model = VggVox()
    input_shape = (299,40,1)
    model = model.res_34(input_shape)
    print(model.summary())