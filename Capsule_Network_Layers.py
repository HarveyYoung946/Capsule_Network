from tensorflow.keras import backend as K
from keras import initializers,layers,regularizers
import tensorflow as tf
#import numpy as np

def squash(vectors,axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors),axis,keepdims=True)
    scale = s_squared_norm / (1+s_squared_norm)/tf.sqrt(s_squared_norm+K.epsilon())
    return tf.multiply(scale,vectors)

class Length(layers.Layer):

    def call(self, inputs,**kwargs):
        return K.sqrt(K.sum(K.square(inputs),-1)+K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(layers.Layer):
    """
       Mask a Tensor with shape=[None, num_capsule, dim_vector]=>[None,10,16] either by the capsule with max length or by an additional
       input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the masked Tensor.
       For example:
           x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
           y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
           out = Mask()(x)  # out.shape=[8, 6]
           # or
           out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list: #inputs=[outputs,true label],and true label is provided with shape=[None,nclass]
            assert len(inputs) == 2
            inputs, mask = inputs
        else:#if no true label,mask by the max length of capsule.mainly used for prediction
            #compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs),-1))
            #generate the mask which is an one-hot code
            #mask.shape=[None,n_class]=[None,num_capsule]
            mask = K.one_hot(indices=K.argmax(x,1),num_classes=x.get_shape().as_list()[1])
        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        t = inputs * K.expand_dims(mask, -1)
        print(['the shape of t',t.shape])
        masked = K.batch_flatten(t)
        print(['the shape of masked',masked])
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple: #true label provided
            return tuple([None,input_shape[0][1]*input_shape[0][2]])
        else: #no lable provided
            return tuple([None,input_shape[1]*input_shape[2]])

class CapsuleLayer(layers.Layer):
    def __init__(self,num_capsule,dim_capsule,routings=3,
                 kernel_initializer='glorot_uniform',**kwargs):

        super(CapsuleLayer,self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        print(['the shape of input_shape',input_shape])
        assert len(input_shape)  >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        #transform matrix
        #num_capsule=10 即要分类的数量 dim_capsule=16 即每个分类向量的大小
        #input_num_capsule=1152   input_dim_capsule=8
        self.W = self.add_weight(shape=[self.num_capsule,self.input_num_capsule,
                                        self.dim_capsule,self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name=self.name+'W')

        self.built = True

    def call(self, inputs, training=None):

      # inputs.shape=[None, input_num_capsule, input_dim_capsule]
      # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
      inputs_expand = K.expand_dims(inputs,1)
      print(['the shape of inputs_expand',inputs_expand.shape])

      #Replicate num_capsule dimension to prepare being multiplied by W
      #inputs_tiled.shape = [None, num_capsule,input_num_capsule, input_dim_capsule]
      input_tiled = K.tile(inputs_expand,[1,self.num_capsule,1,1])
      print(['the shape of iput_tiled', input_tiled.shape])
      print(['the shape of W',self.W.shape])

      # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
      # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
      # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
      # Regard the first two dimensions as `batch` dimension,
      # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
      # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
      inputs_hat = K.map_fn(lambda x: K.batch_dot(x,self.W,[2,3]), elems=input_tiled)
      print(['the shape of input_hat',inputs_hat.shape])

      # Begin: Routing algorithm ---------------------------------------------------------------------#
      # The prior for coupling coefficient, initialized as zeros.
      # b.shape = [None, self.num_capsule, self.input_num_capsule].
      b = tf.zeros(shape=[K.shape(inputs_hat)[0],self.num_capsule,self.input_num_capsule])
      print(['the shape of b',b.shape])

      assert self.routings > 0,'the routings should be > 0'
      for i in range(self.routings):
          #c.shape = [batch_size,num_capsule,input_num_capsule]
          c = tf.nn.softmax(b,axis=1)
          print(['the shape of c',c.shape])
          # c.shape =  [batch_size, num_capsule, input_num_capsule]
          # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
          # The first two dimensions as `batch` dimension,
          # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
          # outputs.shape=[None, num_capsule, dim_capsule]
          outputs = squash(K.batch_dot(c,inputs_hat,[2,2])) #[None,10,16]
          print(['the shape of outputs',outputs.shape])
          if i < self.routings-1:
            # outputs.shape =  [None, num_capsule, dim_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
            # b.shape=[batch_size, num_capsule, input_num_capsule]
            b += K.batch_dot(outputs,inputs_hat,[2,3])

    # End: Routing algorithm
      return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None,self.num_capsule,self.dim_capsule])


def PrimaryCap(inputs,dim_capsule,n_channels,kernel_size,strides,padding):

    output = layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size,strides=strides,padding=padding,name='primarycap_conv2d')(inputs)
    #output = layers.BatchNormalization(momentum=0.9,name='primarycap_bn')(output)
    #output = layers.Activation('relu',name='primarycap_relu')(output)
    #print(output.shape)
    outputs = layers.Reshape(target_shape=[1152,dim_capsule],name='Primarycap_reshape')(output)
    print(outputs.shape)
    return  layers.Lambda(squash,name='primarycap_squash')(outputs)