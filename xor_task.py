import numpy as np
import pandas as pd
import os
import tensorflow as tf
from node_cell import LSTMCell,CTRNNCell,CTLSTM,VanillaRNN,CTGRU,BidirectionalRNN, GRUD, PhasedLSTM, GRUODE
from tqdm import tqdm
import argparse
from ltc.cells.solution import LTCSolution
from ltc.cells.node_solution import NODESolution


class XORData:
    
    def __init__(self,time_major,event_based=True,pad_size=24):
        self.pad_size = pad_size
        self.event_based = event_based
        self._abort_counter=0
        if(not self.load_from_cache()):
            self.create_dataset()

        self.train_elapsed /= self.pad_size
        self.test_elapsed /= self.pad_size

    def load_from_cache(self):
        if(os.path.isfile("dataset/xor_test_y.npy")):
            self.train_events = np.load("dataset/xor_train_events.npy")
            self.train_elapsed = np.load("dataset/xor_train_elapsed.npy")
            self.train_mask = np.load("dataset/xor_train_mask.npy")
            self.train_y = np.load("dataset/xor_train_y.npy")

            self.test_events = np.load("dataset/xor_test_events.npy")
            self.test_elapsed = np.load("dataset/xor_test_elapsed.npy")
            self.test_mask = np.load("dataset/xor_test_mask.npy")
            self.test_y = np.load("dataset/xor_test_y.npy")

            print("train_events.shape: ",str(self.train_events.shape))
            print("train_elapsed.shape: ",str(self.train_elapsed.shape))
            print("train_mask.shape: ",str(self.train_mask.shape))
            print("train_y.shape: ",str(self.train_y.shape))

            print("test_events.shape: ",str(self.test_events.shape))
            print("test_elapsed.shape: ",str(self.test_elapsed.shape))
            print("test_mask.shape: ",str(self.test_mask.shape))
            print("test_y.shape: ",str(self.test_y.shape))
            return True
        return False
    
    def create_event_based_sample(self,rng):
        
        label = 0
        events = np.zeros([self.pad_size,1],dtype=np.float32)
        elapsed = np.zeros([self.pad_size,1],dtype=np.float32)
        mask = np.zeros([self.pad_size],dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0
        length = rng.randint(low=2,high=self.pad_size)
        
        for i in range(length):
            elapsed_counter += 1

            char = int(rng.randint(low=0,high=2))
            label += char
            if(last_char != char):
                events[write_index] = char
                elapsed[write_index] = elapsed_counter
                mask[write_index] = True
                write_index +=1
                elapsed_counter = 0
                if(write_index >= self.pad_size-1):
                    # Enough 1s in this sample, abort
                    self._abort_counter += 1
                    break
            last_char = char
        if(elapsed_counter > 0):
            events[write_index] = char
            elapsed[write_index] = elapsed_counter
            mask[write_index] = True
        label = label % 2
        return events,elapsed,mask,label

    def create_dense_sample(self,rng):
        
        label = 0
        events = np.zeros([self.pad_size,1],dtype=np.float32)
        elapsed = np.zeros([self.pad_size,1],dtype=np.float32)
        mask = np.zeros([self.pad_size],dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0

        length = rng.randint(low=2,high=self.pad_size)
        for i in range(length):
            elapsed_counter += 1

            char = int(rng.randint(low=0,high=2))
            label += char
            events[write_index] = char
            elapsed[write_index] = elapsed_counter
            mask[write_index] = True
            write_index +=1
            elapsed_counter = 0
        label = label % 2
        label2 = int(np.sum(events)) % 2
        assert label == label2
        return events,elapsed,mask,label

    def create_set(self,size,seed):
        rng = np.random.RandomState(seed)
        events_list = []
        elapsed_list = []
        mask_list = []
        label_list = []

        for i in tqdm(range(size)):
            if(self.event_based):
                events,elapsed,mask,label = self.create_event_based_sample(rng)
            else:
                events,elapsed,mask,label = self.create_dense_sample(rng)
            events_list.append(events)
            elapsed_list.append(elapsed)
            mask_list.append(mask)
            label_list.append(label)

        return np.stack(events_list,axis=0),np.stack(elapsed_list,axis=0),np.stack(mask_list,axis=0),np.stack(label_list,axis=0)


    def create_dataset(self):

        print("Transforming training samples")
        self.train_events,self.train_elapsed,self.train_mask,self.train_y = self.create_set(100000,1234984)
        print("Transforming test samples")
        self.test_events,self.test_elapsed,self.test_mask,self.test_y = self.create_set(10000,48736)

        print("train_events.shape: ",str(self.train_events.shape))
        print("train_elapsed.shape: ",str(self.train_elapsed.shape))
        print("train_mask.shape: ",str(self.train_mask.shape))
        print("train_y.shape: ",str(self.train_y.shape))

        print("test_events.shape: ",str(self.test_events.shape))
        print("test_elapsed.shape: ",str(self.test_elapsed.shape))
        print("test_mask.shape: ",str(self.test_mask.shape))
        print("test_y.shape: ",str(self.test_y.shape))

        print("Abort counter: ",str(self._abort_counter))
        os.makedirs("dataset",exist_ok=True)
        np.save("dataset/xor_train_events.npy",self.train_events)
        np.save("dataset/xor_train_elapsed.npy",self.train_elapsed)
        np.save("dataset/xor_train_mask.npy",self.train_mask)
        np.save("dataset/xor_train_y.npy",self.train_y)

        np.save("dataset/xor_test_events.npy",self.test_events)
        np.save("dataset/xor_test_elapsed.npy",self.test_elapsed)
        np.save("dataset/xor_test_mask.npy",self.test_mask)
        np.save("dataset/xor_test_y.npy",self.test_y)

parser = argparse.ArgumentParser()
parser.add_argument('--model',default="ctlstm")
parser.add_argument('--size',default=64,type=int)
parser.add_argument('--epochs',default=200,type=int)
parser.add_argument('--batchsize',default=256,type=int)
parser.add_argument('--lr',default=0.0005,type=float)
parser.add_argument('--dense',action="store_true")
args = parser.parse_args()

if(args.dense):
    data = XORData(time_major=False,event_based=False,pad_size=32)
else:
    data = XORData(time_major=False,event_based=True,pad_size=32)

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

pixel_input = tf.keras.Input(shape=(data.pad_size,1), name='pixel')
time_input = tf.keras.Input(shape=(data.pad_size,1), name='time')
mask_input = tf.keras.Input(shape=(data.pad_size,),dtype=tf.bool, name='mask')

rnn = tf.keras.layers.RNN(cell,time_major=False,return_sequences=False)
dense_layer = tf.keras.layers.Dense(1)

output_states = rnn((pixel_input,time_input),mask=mask_input)
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input,time_input,mask_input],outputs=[y])

model.compile(optimizer=tf.keras.optimizers.RMSprop(args.lr),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])
model.summary()

# Fit model
hist = model.fit(x=(data.train_events,data.train_elapsed,data.train_mask),y=data.train_y,batch_size=args.batchsize,epochs=args.epochs)
# Evaluate model after training
_,best_test_acc = model.evaluate(x=(data.test_events,data.test_elapsed,data.test_mask),y=data.test_y,verbose=2)

# log results
if(args.dense):
    base_path = "results/xor_dense"
else:
    base_path = "results/xor_event"
os.makedirs(base_path,exist_ok=True)
with open("{}/{}_{}.csv".format(base_path,args.model,args.size),"a") as f:
    f.write("{:06f}\n".format(best_test_acc))