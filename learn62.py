
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np

import logging
import time


_word2idx=tf.keras.datasets.imdb.get_word_index()
word2idx={w:i+2 for w,i in _word2idx.items()}
word2idx['<pad>']=0#对应空
word2idx['<start']=1
word2idx['<unk>']=2#unknow word
#idx2word={i:w for w,i in word2idx.item()}

def sort_by_len(x,y):#长度排序
    x,y=np.asarray(x) ,np.asarray(y)
    idx=sorted(range(len(x)),key=lambda i:len(x[i]))
    return x[idx],y[idx]

def dataset(is_training,params):

    _types=(tf.int32,tf.int32)

    if is_training:
        c = np.concatenate((x_trai, y_trai), axis=1)
        ds=tf.data.Dataset.from_tensor_slices(c)
        ds=ds.shuffle(params['num_samples'])
        ds=ds.batch(params['batch_size'])
    else:
        c = np.concatenate((x_tes, y_tes), axis=1)
        ds=tf.data.Dataset.from_tensor_slices(c)
        ds = tf.data.Dataset.from_tensor_slices(c)

        ds = ds.batch(params['batch_size'])
    return ds

class Model(tf.keras.Model):
    def __init__(self,params):
        super().__init__()
        self.embedding=tf.Variable(embedding)

        self.drop1=tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2=tf.keras.layers.Dropout(params['dropout_rate'])

        self.drop4=tf.keras.layers.Dropout(params['dropout_rate'])

        self.rnn1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'],return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))

        self.rnn4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=False))

        self.drop_fc=tf.keras.layers.Dropout(params['dropout_rate'])
        self.fc=tf.keras.layers.Dense(2*params['rnn_units'], tf.nn.relu)
        self.out_linear=tf.keras.layers.Dense(2)
    def call(self,inputs,training=False):
        if inputs.dtype !=tf.int32:
            inputs = tf.cast(inputs,tf.int32)
        batch_sz=tf.shape(inputs)[0]
        rnn_units=2*params['rnn_units']
        x=tf.nn.embedding_lookup(self.embedding,inputs)

        x=self.drop1(x,training=training)
        x=self.rnn1(x)
        x = self.drop2(x, training=training)
        x = self.rnn2(x)

        x = self.drop4(x, training=training)
        x = self.rnn4(x)
        x = self.drop_fc(x, training=training)
        x = self.fc(x)
        x=self.out_linear(x)
        return x

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data()
x_train,y_train=sort_by_len(x_train,y_train)
x_test,y_test=sort_by_len(x_test,y_test)
i=0

#x_trai=np.zeros((25000,100))
#x_tes=np.zeros((25000,100))
x_trai=[[0 for i in range(1500)] for j in range(25000)]#对数据进行处理，超过一定数量的则删除，没超过则补全
x_tes=[[0 for i in range(1500)] for j in range(25000)]
y_trai=[[0 for i in range(2)] for j in range(25000)]
y_tes=[[0 for i in range(2)] for j in range(25000)]
while i<25000:

    j=0
    while j<1500:
        if j<int(len(x_test[i])):
            x_tes[i][j]=x_test[i][j]+2
        else:
            x_tes[i][j]=0
        j+=1
    y_trai[i][0]=y_train[i]
    y_tes[i][0]=y_test[i]
    i+=1
i=0
while i<25000:

    j=0
    while j<1500:
        if j<int(len(x_train[i])):
            x_trai[i][j]=x_train[i][j]+2
        else:
            x_trai[i][j]=0
        j+=1
    i+=1

embedding=np.zeros((len(word2idx),50))

with open('./glove.6B.50d.txt',encoding='utf-8')as f:#进行预处理数据的读取
    count=0
    for i,line in enumerate(f):
        line=line.rstrip()
        sp=line.split(' ')
        word,vec = sp[0],sp[1:]
        if word in word2idx:

            count +=1
            embedding[word2idx[word]]=np.asarray(vec,dtype='float32')
print(count)

print(x_trai[0])
print(len(x_train[24999]))






params={
'num_samples':25000,
'batch_size':32,
'rnn_units':200,
'dropout_rate':0.2,#防止过拟合
'num_patience':3,
'lr':3e-4,

}
def is_descending(history: list):#判断如今精度对比最近精度是否提升
    history = history[-(params['num_patience'] + 1):]
    for i in range(1, len(history)):
        if history[i - 1] <= history[i]:
            return False
    return True




model=Model(params)
model.build(input_shape=(None,None))
decay_lr=tf.optimizers.schedules.ExponentialDecay(params['lr'],1000,0.95)#学习率衰减
optim=tf.optimizers.Adam(params['lr'])
global_step=0

history_acc=[]
best_acc=.0

t0=time.time()
logger=logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

while True:
    for doit in dataset(is_training=True,params=params):

        texts=doit[:,0:1500]

        labels=doit[:,1500]

        with tf.GradientTape() as tape:
            logits=model(texts,training=True)
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss=tf.reduce_mean(loss)

        optim.lr.assign(decay_lr(global_step))
        grads = tape.gradient(loss, model.trainable_variables)

        optim.apply_gradients(zip(grads,model.trainable_variables))

        if global_step%50==0:
            logger.info("Step {} |loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
                global_step,loss.numpy().item(),time.time()-t0,optim.lr.numpy().item()
            ))
            t0=time.time()
        global_step+=1

    m=tf.keras.metrics.Accuracy()

    for doit in dataset(is_training=False, params=params):
        texts=doit[:,0:1500]
        labels=doit[:,1500]
        logits = model(texts, training=False)
        y_pred=tf.argmax(logits,axis=-1)
        m.update_state(y_true=labels, y_pred=y_pred)

    acc=m.result().numpy()
    logger.info("Evaluation:Testing Accuracy: {:.3f}".format(acc))
    history_acc.append(acc)

    if acc>best_acc:
        best_acc=acc
    logger.info("Best Accuracy: {:.3f}".format(best_acc))

    if len(history_acc)>params['num_patience']and is_descending(history_acc):
        logger.info("Testing Accuracy not improved over {} epochs,Early Stop".format(params['num_patience']))
        break
