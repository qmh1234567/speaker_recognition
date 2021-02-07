# proposed attentice Deep CNN
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

class Att_DCNN():
    
    def __init__(self):
      self.WEIGHT_DECAY =  0.00001 
      self.REDUCTION_RATIO = 8
      self.layers_num = [2,3,2]
      
    # 残差块
    def bottle_neck(self,x,outdim,name,strides=(1,1),downsample=False):

        if downsample:
            
            identity = Conv2D(outdim,kernel_size=(1,1),strides=strides,padding='same',name=f'{name}_identity',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x)
            
            identity = BatchNormalization(name=f'{name}_bn0')(identity)
            
        else:
            identity = x

        # conv 1x1
        x = Conv2D(outdim//4,kernel_size=(1,1),strides=(1,1),name=f'{name}_conv1',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn1')(x)
        
        x = Activation('relu',name=f'{name}_relu1')(x)

        # conv 3x3
        x = Conv2D(outdim//4,kernel_size=(3,3),strides=strides,padding='same',name=f'{name}_conv2',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn2')(x)
        
        x = Activation('relu',name=f'{name}_relu2')(x)

        # conv 1x1
        x = Conv2D(outdim,kernel_size=(1,1),strides=(1,1),name=f'{name}_conv3',
        kernel_regularizer= regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn3')(x)

        x = Add(name=f'{name}_scut')([identity,x])
        
        x = Activation('relu',name=f'{name}_relu3')(x)
        
        return x
    
    
    # 提出的残差块
    def resBlock(self,x,outdim,blockNums,name,strides):
        
        # 块中的第一层 需要downsample 和 strides
        x = self.bottle_neck(x,outdim,name=f'{name}_layer0',strides=strides,downsample=True)
        
        # 后面的层数
        for i in range(1,blockNums):
            
            x = self.bottle_neck(x,outdim,name=f'{name}_layer{i}')
            
        return x

    
    # 基线模型
    def baseline_Model(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        x = Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',name='conv1',
        kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x_in)
        
        X = BatchNormalization(name='bn1')(x)
        
        x = Activation('relu',name='relu1')(x)
        
        x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool1')(x)

        x = self.resBlock(x,256,self.layers_num[0],name='block0',strides=(1,1))
        
        x = self.resBlock(x,512,self.layers_num[1],name='block1',strides=(2,2))
        
        x = self.resBlock(x,512,self.layers_num[2],name='block2',strides=(2,2))#1024

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        
        x = Dense(512,name='fc1')(x)#1024
        
        x = BatchNormalization(name='bn_fc1')(x)
        
        x = Activation('relu',name='fc1_relu')(x)
        
        return Model(x_in,x,name='BasicModule')
    
    #  //////////////////////////////////////////////////////////////////////////////////////
    
    
    # CBAM的通道注意力模块
    def channel_attention(self,x,name):
        
        out_dim = x.shape[-1]

        shared_layer_one = Dense(out_dim//self.REDUCTION_RATIO,activation='relu')

        shared_layer_two = Dense(out_dim)


        avg_out = GlobalAveragePooling2D(name=f'{name}_avg_squeeze')(x)
        avg_out = shared_layer_one(avg_out)
        avg_out = shared_layer_two(avg_out)
        # avg_out = Dense(out_dim//reduction_ratio,activation='relu',name=f'{name}_avg_ex1')(avg_out)
        # avg_out = Dense(out_dim,name=f'{name}_avg_ex2')(avg_out)

        max_out = GlobalMaxPool2D(name=f'{name}_max_squeeze')(x)
        max_out = shared_layer_one(max_out)
        max_out = shared_layer_two(max_out)
        # max_out = Dense(out_dim//reduction_ratio,activation='relu',name=f'{name}_max_ex1')(max_out)
        # max_out = Dense(out_dim,name=f'{name}_max_ex2')(max_out)

        out = Add(name=f'{name}_add')([max_out,avg_out])

        out = Activation('sigmoid',name=f'{name}_sigmoid')(out)

        # out = Dense(out_dim,activation='sigmoid',name=f'{name}_scale')(out)

        x = Multiply(name=f'{name}_mul')([out,x])
        
        return x
        
    # 改进的残差块
    def attentive_bottle_neck(self,x,outdim,name,strides=(1,1),downsample=False):

        if downsample:
            
            identity = Conv2D(outdim,kernel_size=(1,1),strides=strides,padding='same',name=f'{name}_identity',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x)
            
            identity = BatchNormalization(name=f'{name}_bn0')(identity)
            
        else:
            identity = x

        # conv 1x1
        x = Conv2D(outdim//4,kernel_size=(1,1),strides=(1,1),name=f'{name}_conv1',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn1')(x)
        
        x = Activation('relu',name=f'{name}_relu1')(x)

        # conv 3x3
        x = Conv2D(outdim//4,kernel_size=(3,3),strides=strides,padding='same',name=f'{name}_conv2',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn2')(x)
        
        x = Activation('relu',name=f'{name}_relu2')(x)

        # conv 1x1
        x = Conv2D(outdim,kernel_size=(1,1),strides=(1,1),name=f'{name}_conv3',
        kernel_regularizer= regularizers.l2(l=self.WEIGHT_DECAY))(x)
        
        x = BatchNormalization(name=f'{name}_bn3')(x)
        
        # 添加CBAM注意力机制模块
        x =  self.channel_attention(x,name=f'{name}_ca')

        x = Add(name=f'{name}_scut')([identity,x])
        
        x = Activation('relu',name=f'{name}_relu3')(x)
        
        return x
    
     # 提出的带注意力机制的残差块
    def attentive_ResBlock(self,x,outdim,blockNums,name,strides):
        
        # 块中的第一层 需要downsample 和 strides
        x = self.attentive_bottle_neck(x,outdim,name=f'{name}_layer0',strides=strides,downsample=True)
        
        # 后面的层数
        for i in range(1,blockNums):
            
            x = self.attentive_bottle_neck(x,outdim,name=f'{name}_layer{i}')
            
        return x
    
    # 提出的模型
    def proposed_model(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        x = Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',name='conv1',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x_in) 
        
        X = BatchNormalization(name='bn1')(x)
        
        x = Activation('relu',name='relu1')(x)

        x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool1')(x)
        
        x = self.attentive_ResBlock(x,256,self.layers_num[0],name='block0',strides=(1,1))  # (1,1)
        
        x = self.attentive_ResBlock(x,512,self.layers_num[1],name='block1',strides=(2,2))
        
        x = self.attentive_ResBlock(x,1024,self.layers_num[2],name='block2',strides=(2,2))

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        
        # x = Dropout(0.5,name=f'dropout')(x)
        
        x = Dense(1024,name='fc1')(x)
        
        x = BatchNormalization(name='bn_fc1')(x)
        
        x = Activation('relu',name='fc1_relu')(x)
        
        # x = Dropout(0.5,name=f'dropout')(x)
        
        return Model(x_in,x,name='ProposedModle')
    
    # ///////////////////////////////////////////////////////
    
    def finetune_Model(self,input_shape):
        x_in = Input(input_shape,name='input')
        def slice(x,index):
            return x[:,index,:]
        x1 = Lambda(slice,arguments={'index':0})(x_in)
        x2 = Lambda(slice,arguments={'index':1})(x_in)
        # 添加计算cosine similarity 的层
        class CosineLayer():
            def __call__(self, x1, x2):
                def _cosine(x):
                    print(len(x))
                    print(x[1].shape)
                    dot1 = K.batch_dot(x[0], x[1], axes=1)  # a*b
                    dot2 = K.batch_dot(x[0], x[0], axes=1) # a*a
                    dot3 = K.batch_dot(x[1], x[1], axes=1) # b*b
                    max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon()) # a*b/(a*a)*(b*b)
                    return dot1 / max_
        
                output_shape = (1,)
                value = Lambda(_cosine,output_shape=output_shape)([x1, x2])
                return value
            
        cosine = CosineLayer()
        x = cosine(x1, x2)
        x = Dense(1,activation='sigmoid',name='logistic')(x)    
        # print(x.shape)
        return Model(x_in,x,name='logistic')  
    
    
    def new_finetune_Model(self,input_shape):
        
        x_in = Input(input_shape,name='input')
        
        x = Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',name='conv1',
            kernel_regularizer=regularizers.l2(l=self.WEIGHT_DECAY))(x_in) 
        
        X = BatchNormalization(name='bn1')(x)
        
        x = Activation('relu',name='relu1')(x)

        x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool1')(x)
        
        x = self.attentive_ResBlock(x,256,self.layers_num[0],name='block0',strides=(1,1))  # (1,1)
        
        x = self.attentive_ResBlock(x,512,self.layers_num[1],name='block1',strides=(2,2))
        
        x = self.attentive_ResBlock(x,1024,self.layers_num[2],name='block2',strides=(2,2))

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        
        x = Dense(1024,name='fc1')(x)
        
        x = BatchNormalization(name='bn_fc1')(x)
        
        x = Activation('relu',name='fc1_relu')(x)
        
        def slice(x,index):
            return x[:,index,:]
        x1 = Lambda(slice,arguments={'index':0})(x)
        x2 = Lambda(slice,arguments={'index':1})(x)
        # 添加计算cosine similarity 的层
        class CosineLayer():
            def __call__(self, x1, x2):
                def _cosine(x):
                    print(len(x))
                    print(x[1].shape)
                    dot1 = K.batch_dot(x[0], x[1], axes=1)  # a*b
                    dot2 = K.batch_dot(x[0], x[0], axes=1) # a*a
                    dot3 = K.batch_dot(x[1], x[1], axes=1) # b*b
                    max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon()) # a*b/(a*a)*(b*b)
                    return dot1 / max_
        
                output_shape = (1,)
                value = Lambda(_cosine,output_shape=output_shape)([x1, x2])
                return value
            
        cosine = CosineLayer()
        x = cosine(x1, x2)
        x = Dense(1,activation='sigmoid',name='logistic')(x)    
        # print(x.shape)
        return Model(x_in,x,name='fine_tune')  
    
if __name__ == "__main__":
    model = Att_DCNN()
    input_shape = (299,40,1)
    model = model.proposed_model(input_shape)
    print(model.summary())
        
        
        