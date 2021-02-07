# proposed SE_ResNet
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

class SE_ResNet():
    
    def __init__(self):
      self.WEIGHT_DECAY =  0.00001 
      self.BLOCK_NUM = 2 # 残差块的数量，可配置
      self.DROPOUT = 0.1
      self.REDUCTION_RATIO = 8
    
    # SE block
    def squeeze_excitation(self,x,name):
        
        out_dim = x.shape[-1]
        
        x = GlobalAveragePooling2D(name=f'{name}_squeeze')(x)
         
        x = Dense(out_dim//self.REDUCTION_RATIO,activation='relu',name=f'{name}_ex0')(x)
    
        x = Dense(out_dim,activation='sigmoid',name=f'{name}_ex1')(x)
    
        return x
    
    
    # 残差块中的卷积块
    def conv_block(self,x,filters,kernel_size,stride,name,stage,i,padding='same'):
        
        x = Conv2D(filters,kernel_size,strides=stride,padding=padding,name=f'{name}_conv{stage}_{i}',
            kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn{stage}_{i}')(x)
        
        if stage != 'c':  # 最后一个卷积块暂时不用激活函数
            x = ELU(name=f'{name}_relu{stage}_{i}')(x)
      
        return x 
 
    
    # 提出的残差块结构
    def residual_block(self,x,outdim,stride,name):
        
        shortcut = Conv2D(outdim,kernel_size=(1,1),strides=stride,name=f'{name}_scut_conv',
            kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)
        
        for i in range(self.BLOCK_NUM):
            
            if i>0:
                stride = 1
                x = Dropout(self.DROPOUT,name=f'{name}_drop{i-1}')(x)
            
            x = self.conv_block(x,outdim//4,(1,1),stride,name,'a',i,padding='same')
            x = self.conv_block(x,outdim//4,(3,3),(1,1),name,'b',i,padding='same')
            x = self.conv_block(x,outdim,(1,1),(1,1),name,'c',i,padding='same')
            # x = ELU(name=f'{name}_relu{i}')(x)

        # add SE block
        x = Multiply(name=f'{name}_scale')([x,self.squeeze_excitation(x,name)])
        
        x = Add(name=f'{name}_scut')([shortcut,x])
        
        x = ELU(name=f'{name}_relu')(x)
        
        return x 
      
    # SE_ResNet model
    def se_resNet(self,input_shape):
  
        x_in =Input(input_shape,name='input')
        
        x = Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv1',kernel_regularizer = regularizers.l2(l=self.WEIGHT_DECAY))(x_in)
        
        x = BatchNormalization(name='bn1')(x)
        
        x = ELU(name=f'relu1')(x)

        x = MaxPool2D((2,2),strides=(2,2),padding='same',name='pool1')(x)

        x = self.residual_block(x,outdim=256,stride=(2,2),name='block2')
        
        x = self.residual_block(x,outdim=256,stride=(2,2),name='block3')
        
        x = self.residual_block(x,outdim=512,stride=(2,2),name='block6')

        x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='average')(x)

        x = Dense(512,name='fc1')(x)
        
        x = BatchNormalization(name='bn_fc1')(x)
        
        x = ELU(name=f'relu_fc1')(x)
        
        model = Model(inputs=[x_in],outputs=[x],name='SEResNet')
        
        return model

if __name__ == "__main__":
    model = SE_ResNet()
    input_shape = (299,40,1)
    model = model.se_resNet(input_shape)
    print(model.summary())