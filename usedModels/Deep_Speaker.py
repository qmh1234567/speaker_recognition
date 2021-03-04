# deep speaker
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


class DeepSpeaker():
    def __init__(self):
      self.WEIGHT_DECAY =  0.00001 
    #   self.age = age
    
    # 激活函数
    def clipped_relu(self,inputs):
        return Lambda(lambda y: K.minimum(K.maximum(y,0),20))(inputs)

    # 残差块
    def identity_block(self,x_in,kernel_size,filters,name):
        
        x = Conv2D(filters,kernel_size=kernel_size,strides=(1,1),padding='same',
                kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY),name=f'{name}_conva')(x_in)

        x = BatchNormalization(name=f'{name}_bn1')(x)
        
        x = self.clipped_relu(x)
        
        x = Conv2D(filters,kernel_size=kernel_size,strides=(1,1),padding='same',
                kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY),name=f'{name}_convb')(x)
        
        x = BatchNormalization(name=f'{name}_bn2')(x)
        
        x =  Add(name=f'{name}_add')([x,x_in])
        
        x = self.clipped_relu(x)
        
        return x
    
    # 卷积和残差块
    def conv_and_res_block(self,x_in,filters):
        
        x = Conv2D(filters,kernel_size=(5,5),strides=(2,2),
        padding='same',kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY),
        name=f'conv_{filters}-s')(x_in)
        
        x = BatchNormalization(name=f'conv_{filters}-s_bn')(x)
        
        x = self.clipped_relu(x)
        
        for i in range(3):
            x = self.identity_block(x,kernel_size=(3,3),filters=filters,name=f'res{filters}_{i}')
            
        return x
    
    # deep speaker模型
    def deep_speaker_model(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        x = self.conv_and_res_block(x_in,64)
        
        x = self.conv_and_res_block(x,128)
        
        x = self.conv_and_res_block(x,256)
        
        x = self.conv_and_res_block(x,512)
        
        # average
        # x = Lambda(lambda y:K.mean(y,axis=[1,2]),name='avgpool')(x)
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        
        # affine
        x = Dense(512,name='fc1')(x)
        
        #no BN
        
        x = BatchNormalization(name='bn_fc1')(x)  
        
        x = Activation('relu',name='fc1_relu')(x)
        
        # 归一化
        # x = Lambda(lambda y:K.l2_normalize(y,axis=1),name='ln')(x)
        
        model = Model(inputs=[x_in],outputs=[x],name='deepspeaker')
        
        return model
        

if __name__ == "__main__":
    model = DeepSpeaker()
    input_shape = (299,40,1)
    model = model.deep_speaker_model(input_shape)
    print(model.summary())