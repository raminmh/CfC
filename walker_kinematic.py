import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from node_cell import LSTMCell,CTRNNCell,CTLSTM,VanillaRNN,CTGRU,BidirectionalRNN, GRUD, PhasedLSTM, GRUODE
from tqdm import tqdm
import argparse
from ltc.cells.solution import LTCSolution
from ltc.cells.node_solution import NODESolution


class ImitationData:

    def __init__(self,seq_len):
        self.seq_len = seq_len
        all_files = sorted([os.path.join("data/walker",d) for d in os.listdir("data/walker") if d.endswith(".npy")])

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15*len(all_files))
        valid_n = int((0.15+0.1)*len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x,train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x,train_t,train_y = self.perturb_sequences(train_x,train_t,train_y)
        valid_x,valid_t,valid_y = self.perturb_sequences(valid_x,valid_t,valid_y)
        test_x,test_t,test_y = self.perturb_sequences(test_x,test_t,test_y)

        self.train_x,self.train_times,self.train_y = self.align_sequences(train_x,train_t,train_y)
        self.valid_x,self.valid_times,self.valid_y = self.align_sequences(valid_x,valid_t,valid_y)
        self.test_x,self.test_times,self.test_y = self.align_sequences(test_x,test_t,test_y)
        self.input_size = self.train_x.shape[-1]

        print("train_times: ",str(self.train_times.shape))
        print("train_x: ",str(self.train_x.shape))
        print("train_y: ",str(self.train_y.shape))

    def align_sequences(self,set_x,set_t,set_y):
        
        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(0,seq_y.shape[0]-self.seq_len,self.seq_len//4):
                x.append(seq_x[t:t+self.seq_len])
                times.append(seq_t[t:t+self.seq_len])
                y.append(seq_y[t:t+self.seq_len])
        
        return (
            np.stack(x,axis=0),
            np.expand_dims(np.stack(times,axis=0),axis=-1),
            np.stack(y,axis=0),
        )


    def perturb_sequences(self,set_x,set_t,set_y):
        
        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x,new_times = [], []
            new_y = []

            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if(self.rng.rand()<0.9):
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip=0

            x.append(np.stack(new_x,axis=0))
            times.append(np.stack(new_times,axis=0))
            y.append(np.stack(new_y,axis=0))
        
        return x,times,y


    def _load_files(self,files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:
           
            arr = np.load(f)
            x_state = arr[:-1,:].astype(np.float32)
            y = arr[1:,:].astype(np.float32)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            print("Loaded file '{}' of length {:d}".format(f,x_state.shape[0]))
        return all_x,all_t,all_y



parser = argparse.ArgumentParser()
parser.add_argument('--model',default="solution_ltc")
parser.add_argument('--size',default=32,type=int)
parser.add_argument('--epochs',default=200,type=int)
parser.add_argument('--batchsize',default=24,type=int)
parser.add_argument('--lr',default=0.01,type=float)
args = parser.parse_args()

data = ImitationData(seq_len=64)

if(args.model == "lstm"):
    cell = LSTMCell(units=args.size)
elif(args.model == "ctrnn"):
    cell = CTRNNCell(units=args.size,num_unfolds=3,method="rk4")
elif(args.model == "node"):
    cell = CTRNNCell(units=args.size,num_unfolds=3,method="rk4",tau=0)
elif(args.model == "ctlstm"):
    cell = CTLSTM(units=args.size)
elif(args.model == "ctgru"):
    cell = CTGRU(units=args.size)
elif(args.model == "vanilla"):
    cell = VanillaRNN(units=args.size)
elif(args.model == "bidirect"):
    cell = BidirectionalRNN(units=args.size)
elif(args.model == "grud"):
    cell = GRUD(units=args.size)
elif(args.model == "phased"):
    cell = PhasedLSTM(units=args.size)
elif(args.model == "gruode"):
    cell = GRUODE(units=args.size)
elif (args.model == "solution_ltc"):
    cell = LTCSolution(args.size)
elif (args.model == "solution_node"):
    cell = NODESolution(args.size)
else:
    raise ValueError("Unknown model type '{}'".format(args.model))

signal_input = tf.keras.Input(shape=(data.seq_len,data.input_size), name='robot')
time_input = tf.keras.Input(shape=(data.seq_len,1), name='time')

rnn = tf.keras.layers.RNN(cell,time_major=False,return_sequences=True)

output_states = rnn((signal_input,time_input))
y = tf.keras.layers.Dense(data.input_size)(output_states)

model = tf.keras.Model(inputs=[signal_input,time_input],outputs=[y])

model.compile(optimizer=tf.keras.optimizers.RMSprop(args.lr),loss=tf.keras.losses.MeanSquaredError())
model.summary()

hist = model.fit(
    x=(data.train_x,data.train_times),
    y=data.train_y,batch_size=args.batchsize,epochs=args.epochs,
    validation_data=((data.valid_x,data.valid_times),data.valid_y),
    callbacks=[tf.keras.callbacks.ModelCheckpoint("/tmp/checkpoint",save_best_only=True,save_weights_only=True,mode="min")])

# Restore checkpoint with lowest validation MSE
model.load_weights("/tmp/checkpoint")
best_test_loss = model.evaluate(x=(data.test_x,data.test_times),y=data.test_y,verbose=2)
print("Best test loss: {:0.3f}".format(best_test_loss))

# Log result in file
base_path = "results/walker"
os.makedirs(base_path,exist_ok=True)
with open("{}/{}_{}.csv".format(base_path,args.model,args.size),"a") as f:
    f.write("{:06f}\n".format(best_test_loss))