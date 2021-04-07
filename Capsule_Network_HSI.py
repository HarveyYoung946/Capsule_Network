from keras import layers,models,optimizers,datasets,callbacks,regularizers
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
import math
import pandas
import numpy as np
from  keras.utils import to_categorical
from Capsule_Network_Layers import PrimaryCap,CapsuleLayer,Length,Mask

def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    print(num)
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def margin_loss(y_true,y_pred):
    """
       Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
       :param y_true: [None, n_classes]
       :param y_pred: [None, num_capsule]
       :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0.,0.9-y_pred)) + 0.5 * (1-y_true) * K.square(K.maximum(0.,y_pred-0.1))
    return  K.mean(K.sum(L,1))


def CapsNet(input_shape,n_class,num_routing):

    #Input
    x = layers.Input(shape=input_shape)
    #conv_1
    conv_1 = layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='valid')(x)
    conv_1 = layers.BatchNormalization(momentum=0.9)(conv_1)
    conv_1 = layers.Activation('relu')(conv_1)
    # conv_2
    conv_2 = layers.Conv2D(filters=64,kernel_size=3,strides=2,padding='valid')(conv_1)
    conv_2 = layers.BatchNormalization(momentum=0.9)(conv_2)
    conv_2 = layers.Activation('relu')(conv_2)

    primarycaps = PrimaryCap(conv_2, dim_capsule=8, n_channels=32, kernel_size=7, strides=1, padding='valid')

    digtalcaps = CapsuleLayer(num_capsule=n_class,dim_capsule=16,routings=num_routing,name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    outcaps = Length(name='outcaps')(digtalcaps)

    #decoder network
    label = layers.Input(shape=(n_class,))
    masked_by_label = Mask()([digtalcaps,label]) #The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digtalcaps) # Mask using the capsule with maximal length. For prediction
    #print(masked.shape)

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512,activation='relu',input_dim=16*n_class,kernel_regularizer=regularizers.l2(0.01)))
    decoder.add(layers.Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    decoder.add(layers.Dense(np.prod(input_shape),activation='sigmoid')) #np.prod为联乘函数，即将矩阵中的元素相乘
    decoder.add(layers.Reshape(target_shape=input_shape,name='out_reconstruction'))

    #models for training and evaluation(prediction)
    train_model = models.Model([x,label],[outcaps,decoder(masked)])
    eval_model = models.Model(x,[outcaps,decoder(masked)])

    """
    #manipulate model
    noise = layers.Input(shape=(n_class,16))
    print([noise.shape,digtalcaps.shape])
    noise_digitcaps = layers.Add()([digtalcaps,noise])
    masked_noised_y = Mask()([noise_digitcaps,label])
    manipulate_model = models.Model([x,label,noise],decoder(masked_noised_y))
    """


    return train_model,eval_model




def LoadData():

  (x_train,y_train),(x_val,y_val) = datasets.mnist.load_data()
  #print(x_train.shape)
  x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.
  x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255.
  y_train = to_categorical(y_train)
  y_val = to_categorical(y_val)
  #print(y_train.shape)
  '''
  train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
  train_db = train_db.shuffle(10000).batch(batchsz)
  test_db =  tf.data.Dataset.from_tensor_slices((x_val,y_val))
  test_db = test_db.batch(batchsz)
  '''
  return (x_train,y_train),(x_val,y_val)

def train(model,data,args):

    (x_train, y_train), (x_val, y_val) = data

    #callbacks
    log = callbacks.CSVLogger(args.save_dir+'/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir+'/logs',update_freq=args.batchsz,histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir+'/weights-{epoch:02d}.h5',monitor='val_capnet_acc',save_best_only=True,save_weights_only=True,verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda  epoch: args.lr*(args.lr_decay ** epoch))


    lr = 1e-3
    model.compile(optimizer=optimizers.Adam(lr=lr),
                           loss=[margin_loss, 'mse'],
                           metrics={'outcaps': 'accuracy'})

    # training without data augmentation
    # verbose：该参数的值控制日志显示的方式; verbose = 0 不在标准输出流输出日志信息; verbose = 1 输出进度条记录; verbose = 2 每个epoch输出一行记录 注意： 默认为 1
    # 输入与模型的train_model对应，即 train_model = models.Model([x,label],[outcaps,decoder(masked)])
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batchsz, epochs=args.epochs,
               validation_data=[[x_val, y_val], [y_val, x_val]], shuffle=True, verbose=2,
               callbacks=[log,tb,checkpoint,lr_decay])

    model.save_weights(args.save_dir+'/training_model.h5')
    print('trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    plot_log(args.save_dir+'/log.csv',show=True)
    return model

def test(model,data,args):
    t = 5
    x_val,y_val = data
    x_pred, x_recons = model.predict(x_val, batch_size=10)
    print('-' * 30 + 'Begin:test' + '-' * 30)
    x_label = np.sum(np.argmax(x_pred, 1) == np.argmax(y_val, 1)) / y_val.shape[0]
    print('test accurate', x_label)
    print(['the shape of the x_val', x_val.shape])

    # np.concatenate 拼接矩阵
    img = combine_images(np.concatenate([x_val[:50], x_recons[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir+"/real and recons%d.png"%t)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':

    import argparse
    #set the hyper parameters
    parser = argparse.ArgumentParser(description='Capsule Network on MNinst')
    parser.add_argument('--save_dir',default='./capsule_result')
    parser.add_argument('--batchsz',default=20,type=int)
    parser.add_argument('--epochs',default=1,type=int)
    parser.add_argument('--num_routing',default=5,type=int)
    parser.add_argument('--n_class',default=10,type=int)
    parser.add_argument('--lr',default=1e-3,type=int)
    parser.add_argument('--lr_decay',default=0.9,type=float,help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-t','--testing',action='store_true',help='test the trained model on testing dataset')
    parser.add_argument('--debug',action='store_true',help='Save weights by TensorBoard')
    parser.add_argument('-w','--weights',default=None,help='The path of the saved weights')
    args = parser.parse_args()
    print(args)

    (x_train,y_train),(x_val,y_val) = LoadData()

#define model

    training_model,evalidation_model = CapsNet(input_shape=[28,28,1],n_class=args.n_class,num_routing=args.num_routing)
    training_model.summary()

    if args.weights is not None: #init the model with provided one
        training_model.load_weights(args.weights)
    if not args.testing:
        #train model
        train(model=training_model, data=((x_train,y_train),(x_val,y_val)), args=args)
    else:
        if args.weights is None:
            print('No weights are provided')
        #test and reconstruction
        test(model=evalidation_model,data=(x_val,y_val),args=args)
