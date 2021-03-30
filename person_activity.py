import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import argparse
from node_cell import LSTMCell,CTRNNCell,CTLSTM,VanillaRNN,CTGRU,BidirectionalRNN, GRUD, PhasedLSTM, GRUODE
from ltc.cells.solution import LTCSolution
from ltc.cells.node_solution import NODESolution


class_map = {
    'lying down': 0,
    'lying': 0,
    'sitting down': 1,
    'sitting': 1,
    'standing up from lying': 2,
    'standing up from sitting': 2,
    'standing up from sitting on the ground': 2,
    "walking": 3,
    "falling": 4,
    'on all fours': 5,
    'sitting on the ground': 6,
}

sensor_ids = {
    "010-000-024-033":0,
    "010-000-030-096":1,
    "020-000-033-111":2,
    "020-000-032-221":3
}

def one_hot(x,n):
    y = np.zeros(n,dtype=np.float32)
    y[x] = 1
    return y

def load_crappy_formated_csv():

    all_x = []
    all_y = []
    all_t = []

    series_x = []
    series_t = []
    series_y = []

    all_feats = []
    all_elapsed = []
    last_millis = None
    if(not os.path.isfile("data/person/ConfLongDemo_JSI.txt")):
        print("ERROR: File 'data/person/ConfLongDemo_JSI.txt' not found")
        print("Please execute the command")
        print("source download_dataset.sh")
        import sys
        sys.exit(-1)
    with open("data/person/ConfLongDemo_JSI.txt","r") as f:
        current_person = "A01"

        for line in f:
            arr = line.split(",")
            if(len(arr)<6):
                break
            if(arr[0] != current_person):
                # Enque and reset
                series_x = np.stack(series_x,axis=0)
                series_t = np.stack(series_t,axis=0)
                series_y = np.array(series_y,dtype=np.int32)
                all_x.append(series_x)
                all_t.append(series_t)
                all_y.append(series_y)
                last_millis = None
                series_x = []
                series_y = []
                series_t = []

            millis = np.int64(arr[2])/(100*1000)
            # 100ms will be normalized to 1.0
            millis_mapped_to_1 = 10.0
            if(last_millis is None):
                elasped_sec = 0.05
            else:
                elasped_sec = float(millis-last_millis)/1000.0
            elasped  = elasped_sec*1000/millis_mapped_to_1

            last_millis = millis
            all_elapsed.append(elasped)
            current_person = arr[0]
            sensor_id = sensor_ids[arr[1]]
            label_col = class_map[arr[7].replace("\n","")]
            feature_col_2 = np.array(arr[4:7],dtype=np.float32)
            # Last 3 entries of the feature vector contain sensor value

            # First 4 entries of the feature vector contain sensor ID
            feature_col_1 = np.zeros(4,dtype=np.float32)
            feature_col_1[sensor_id] = 1

            feature_col = np.concatenate([feature_col_1,feature_col_2])
            series_x.append(feature_col)
            series_t.append(elasped)
            all_feats.append(feature_col)
            series_y.append(label_col)

    all_feats = np.stack(all_feats,axis=0)
    print("all_feats.shape: ",str(all_feats.shape))
    all_elapsed = np.stack(all_elapsed,axis=0)
    print("all_elapsed.mean",str(np.mean(all_elapsed)))
    print("all_elapsed.med ",str(np.median(all_elapsed)))
    
    # No normalization
    # all_mean = np.mean(all_feats,axis=0)
    # all_std = np.std(all_feats,axis=0)
    # all_mean[3:] = 0
    # all_std[3:] = 1
    # print("all_mean: ",str(all_mean))
    # print("all_std: ",str(all_std))
    # for i in range(len(all_x)):
    #     all_x[i] -= all_mean
    #     all_x[i] /= all_std

    return all_x,all_t,all_y


def cut_in_sequences(all_x,all_t,all_y,seq_len,inc=1):

    sequences_x = []
    sequences_t = []
    sequences_y = []

    for i in range(len(all_x)):
        x,t,y = all_x[i],all_t[i],all_y[i]

        for s in range(0,x.shape[0] - seq_len,inc):
            start = s
            end = start+seq_len
            sequences_x.append(x[start:end])
            sequences_t.append(t[start:end])
            sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=0),np.stack(sequences_t,axis=0).reshape([-1,seq_len,1]),np.stack(sequences_y,axis=0)

class PersonData:

    def __init__(self,seq_len=32):

        self.seq_len = seq_len
        self.num_classes = 7
        all_x,all_t,all_y = load_crappy_formated_csv()
        all_x,all_t,all_y = cut_in_sequences(all_x,all_t,all_y,seq_len=seq_len,inc=seq_len//2)

        print("all_x.shape: ",str(all_x.shape))
        print("all_t.shape: ",str(all_t.shape))
        print("all_y.shape: ",str(all_y.shape))
        total_seqs = all_x.shape[0]
        print("Total number of sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(98841).permutation(total_seqs)
        test_size = int(0.2*total_seqs)

        self.test_x = all_x[permutation[:test_size]]
        self.test_y = all_y[permutation[:test_size]]
        self.test_t = all_t[permutation[:test_size]]
        self.train_x = all_x[permutation[test_size:]]
        self.train_t = all_t[permutation[test_size:]]
        self.train_y = all_y[permutation[test_size:]]

        self.feature_size = int(self.train_x.shape[-1])

        print("train_x.shape: ",str(self.train_x.shape))
        print("train_t.shape: ",str(self.train_t.shape))
        print("train_y.shape: ",str(self.train_y.shape))
        print("Total number of train sequences: {}".format(self.train_x.shape[0]))
        print("Total number of test  sequences: {}".format(self.test_x.shape[0]))



parser = argparse.ArgumentParser()
parser.add_argument('--model',default="solution_node")
parser.add_argument('--size',default=64,type=int)
parser.add_argument('--batchsize',default=1024,type=int)
parser.add_argument('--epochs',default=200,type=int)
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--dense',action="store_true")
args = parser.parse_args()

data = PersonData()

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
elif(args.model == "solution_ltc"):
    cell = LTCSolution(args.size)
elif (args.model == "solution_node"):
    cell = NODESolution(args.size)
else:
    raise ValueError("Unknown model type '{}'".format(args.model))

pixel_input = tf.keras.Input(shape=(data.seq_len,data.feature_size), name='features')
time_input = tf.keras.Input(shape=(data.seq_len,1), name='time')


rnn = tf.keras.layers.RNN(cell,time_major=False,return_sequences=True)
dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(data.num_classes))

output_states = rnn((pixel_input,time_input))
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input,time_input],outputs=[y])

model.compile(optimizer=tf.keras.optimizers.RMSprop(args.lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()

# Fit and evaluate
hist = model.fit(x=(data.train_x,data.train_t),y=data.train_y,batch_size=args.batchsize,epochs=args.epochs)
_,best_test_acc = model.evaluate(x=(data.test_x,data.test_t),y=data.test_y,verbose=2)

# log results
base_path = "results/person_activity"
os.makedirs(base_path,exist_ok=True)
with open("{}/{}_{}.csv".format(base_path,args.model,args.size),"a") as f:
    f.write("{:06f}\n".format(best_test_acc))