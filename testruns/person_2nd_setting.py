import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import time

from sklearn import model_selection


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
} #11 to 7

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

def save_record(series_x,series_y,all_x,all_y):

    series_x = np.stack(series_x,axis=0)
    series_y = np.array(series_y,dtype=np.int32)
    slide = 25
    seq_len = 50
    add_len = 0
    for i in range(0,series_x.shape[0]-seq_len,slide):
        add_len += 1

        part_x = series_x[i:i+seq_len]
        part_y = series_y[i:i+seq_len]

        all_x.append(part_x)
        all_y.append(part_y)


def load_crappy_formated_csv():

    all_x = []
    all_y = []

    series_x = []
    series_y = []

    with open("data/person/ConfLongDemo_JSI.txt","r") as f:
        current_person = None

        first_tp = None
        for line in f:
            arr = line.split(",")
            time = float(arr[2])

            if(len(arr)<6):
                break
            if(arr[0] != current_person):
                # Enque and reset
                if(not current_person is None):
                    save_record(series_x,series_y,all_x,all_y)
                first_tp = time
                time = round((time - first_tp)/ 10**5)
                prev_time = time
                series_x = []
                series_y = []

            current_person = arr[0]
            sensor_id = sensor_ids[arr[1]]
            label_col = class_map[arr[7].replace("\n","")]
            feature_col_2 = np.array(arr[4:7],dtype=np.float32)

            feature_col_1 = np.zeros(4,dtype=np.float32)
            feature_col_1[sensor_id] = 1

            feature_col = np.concatenate([feature_col_1,feature_col_2])
            time = round((time - first_tp)/ 10**5)
            if(time != prev_time):
                series_x.append(feature_col)
                series_y.append(label_col)
                prev_time = time

        save_record(series_x,series_y,all_x,all_y)

    print("All item: ",str(len(all_x)))
    return all_x,all_y


def cut_in_sequences(all_x,all_y):

    sequences_x = []
    sequences_y = []
    print("Cut in sequences: ",str(len(all_x)),", ",str(len(all_y)))
    for i in range(len(all_x)):
        x,y = all_x[i],all_y[i]
        x1,y1 = x[:25],y[:25]
        x2,y2 = x[25:],y[25:]
        sequences_x.append(x1)
        sequences_x.append(x2)
        sequences_y.append(y1)
        sequences_y.append(y2)

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

class PersonData:

    def __init__(self):
        
        all_x,all_y = load_crappy_formated_csv()
        train_x,test_x,train_y,test_y = model_selection.train_test_split(all_x,all_y, train_size= 0.8, 
	random_state = 42, shuffle = True)
        # Now we have the exact same train-test split as Duvenaud

        train_x,train_y = cut_in_sequences(train_x,train_y)
        test_x,test_y = cut_in_sequences(test_x,test_y)

        total_seqs = train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))

        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1*total_seqs)

        self.valid_x = train_x[:,:valid_size]
        self.valid_y = train_y[:,:valid_size]
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_x[:,valid_size:]
        self.train_y = train_y[:,valid_size:]

        print("self.train_x: ",str(self.train_x.shape))
        print("self.train_y: ",str(self.train_y.shape))
        print("self.valid_x: ",str(self.valid_x.shape))
        print("self.valid_y: ",str(self.valid_y.shape))
        print("self.test_x: ",str(self.test_x.shape))
        print("self.test_y: ",str(self.test_y.shape))
        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            yield (batch_x,batch_y)

class PersonModel:

    def __init__(self,model_type,model_size,sparsity_level=0.0):
        self.model_type = model_type
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,4+3])
        self.target_y = tf.placeholder(dtype=tf.int32,shape=[None,None])

        self.model_size = model_size
        learning_rate = args.lr # LTC needs a higher learning rate
        head = self.x
        if(model_type == "lstm"):
            # unstacked_signal = tf.unstack(x,axis=0)
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())

        self.y = tf.layers.Dense(7,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        # config = tf.ConfigProto()
        # config.intra_op_parallelism_threads = 4
        # config.inter_op_parallelism_threads = 1
        # self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # self.result_file = os.path.join("results","person","{}_{}_{:02d}.csv".format(model_type,model_size,int(100*self.sparsity_level)))
        lr = int(learning_rate*10000)
        bs = args.batch_size
        self.result_file = os.path.join("results","duvenaud","{}_{}_lr{}_bs{}.csv".format(model_type,model_size,lr,bs))
        if(not os.path.exists("results/duvenaud")):
            os.makedirs("results/duvenaud")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","duvenaud","{}".format(model_type))
        if(not os.path.exists("tf_sessions/duvenaud")):
            os.makedirs("tf_sessions/duvenaud")
            
        self.saver = tf.train.Saver()

    def get_sparsity_ops(self):
        tf_vars = tf.trainable_variables()
        op_list = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            if(v.name.startswith("rnn")):
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    continue
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    continue
                op_list.append(self.sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        v_assign_op = tf.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        return v_assign_op

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):

        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        print("Entering training loop")
        for e in range(epochs):
            start_time = time.time()
            if(e%log_period == 0):
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_y})
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_y in gesture_data.iterate_train(batch_size=args.batch_size):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                elapsed = time.time()-start_time
                print("Epochs {:03d} ({:0.1f} s), train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,elapsed,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=64,type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--sparsity',default=0.0,type=float)
    args = parser.parse_args()

    person_data = PersonData()
    model = PersonModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity)

    model.fit(person_data,epochs=args.epochs,log_period=args.log)



