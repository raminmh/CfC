import numpy as np
import pandas as pd
import os
import tensorflow as tf
from node_cell import LSTMCell,CTRNNCell,CTLSTM,VanillaRNN,CTGRU,BidirectionalRNN, GRUD, PhasedLSTM, GRUODE
from tqdm import tqdm
import argparse
from ltc.cells.solution import LTCSolution
from ltc.cells.node_solution import NODESolution


class ETSMnistData:
    
    def __init__(self,time_major,pad_size=256):
        self.threshold = 128
        self.pad_size = pad_size
        
        if(not self.load_from_cache()):
            self.create_dataset()
            
        self.train_elapsed /= self.pad_size
        self.test_elapsed /= self.pad_size

    def load_from_cache(self):
        if(os.path.isfile("dataset/test_mask.npy")):
            self.train_events = np.load("dataset/train_events.npy")
            self.train_elapsed = np.load("dataset/train_elapsed.npy")
            self.train_mask = np.load("dataset/train_mask.npy")
            self.train_y = np.load("dataset/train_y.npy")

            self.test_events = np.load("dataset/test_events.npy")
            self.test_elapsed = np.load("dataset/test_elapsed.npy")
            self.test_mask = np.load("dataset/test_mask.npy")
            self.test_y = np.load("dataset/test_y.npy")

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
    
    def transform_sample(self,x):
        x = x.flatten()
        
        events = np.zeros([self.pad_size,1],dtype=np.float32)
        elapsed = np.zeros([self.pad_size,1],dtype=np.float32)
        mask = np.zeros([self.pad_size],dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0
        for i in range(x.shape[0]):
            elapsed_counter += 1
            char = int(x[i]>self.threshold)
            if(last_char != char):
                events[write_index] = char
                elapsed[write_index] = elapsed_counter
                mask[write_index] = True
                write_index +=1
                if(write_index >= self.pad_size):
                    # Enough 1s in this sample, abort
                    self._abort_counter += 1
                    break
                elapsed_counter = 0
            last_char = char
        self._all_lenghts.append(write_index)
        return events,elapsed,mask

    def transform_array(self,x):
        events_list = []
        elapsed_list = []
        mask_list = []

        for i in tqdm(range(x.shape[0])):
            events,elapsed,mask = self.transform_sample(x[i])
            events_list.append(events)
            elapsed_list.append(elapsed)
            mask_list.append(mask)

        return np.stack(events_list,axis=0),np.stack(elapsed_list,axis=0),np.stack(mask_list,axis=0)

    def create_dataset(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

        self._all_lenghts = []
        self._abort_counter = 0

        train_x = train_x.reshape([-1,28*28])
        test_x = test_x.reshape([-1,28*28])

        self.train_y = train_y
        self.test_y = test_y

        print("Transforming training samples")
        self.train_events,self.train_elapsed,self.train_mask = self.transform_array(train_x)
        print("Transforming test samples")
        self.test_events,self.test_elapsed,self.test_mask = self.transform_array(test_x)

        print("Average time-series length: {:0.2f}".format(np.mean(self._all_lenghts)))
        print("Abort counter: ",str(self._abort_counter))
        os.makedirs("dataset",exist_ok=True)
        np.save("dataset/train_events.npy",self.train_events)
        np.save("dataset/train_elapsed.npy",self.train_elapsed)
        np.save("dataset/train_mask.npy",self.train_mask)
        np.save("dataset/train_y.npy",self.train_y)

        np.save("dataset/test_events.npy",self.test_events)
        np.save("dataset/test_elapsed.npy",self.test_elapsed)
        np.save("dataset/test_mask.npy",self.test_mask)
        np.save("dataset/test_y.npy",self.test_y)

parser = argparse.ArgumentParser()
parser.add_argument('--model',default="solution_ltc")
parser.add_argument('--size',default=64,type=int)
parser.add_argument('--epochs',default=200,type=int)
parser.add_argument('--batchsize',default=128,type=int)
parser.add_argument('--lr',default=0.0005,type=float)
args = parser.parse_args()

data = ETSMnistData(time_major=False)

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
dense_layer = tf.keras.layers.Dense(10)

output_states = rnn((pixel_input,time_input),mask=mask_input)
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input,time_input,mask_input],outputs=[y])

model.compile(optimizer=tf.keras.optimizers.RMSprop(args.lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()
# Other possibility: use best test accuracy achieved during training
# hist = model.fit(x=(data.train_events,data.train_elapsed,data.train_mask),y=data.train_y,batch_size=128,epochs=args.epochs,validation_data=((data.test_events,data.test_elapsed,data.test_mask),data.test_y))
# test_accuracies = hist.history["val_sparse_categorical_accuracy"]
# best_test_acc = np.max(test_accuracies)

# Fit and evaluate
hist = model.fit(x=(data.train_events,data.train_elapsed,data.train_mask),y=data.train_y,batch_size=args.batchsize,epochs=args.epochs)

_,best_test_acc = model.evaluate(x=(data.test_events,data.test_elapsed,data.test_mask),y=data.test_y,verbose=2)

os.makedirs("results/smnist",exist_ok=True)
with open("results/smnist/{}_{}.csv".format(args.model,args.size),"a") as f:
    f.write("{:06f}\n".format(best_test_acc))