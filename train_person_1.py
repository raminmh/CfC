import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU
import ltc
from ltc.cells.solution import LTCSolution
from ltc.cells.node_solution import NODESolution
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from sklearn import model_selection
import argparse
import time


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
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3
}


def one_hot(x, n):
    y = np.zeros(n, dtype=np.float32)
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
    if (not os.path.isfile("data/person/ConfLongDemo_JSI.txt")):
        print("ERROR: File 'data/person/ConfLongDemo_JSI.txt' not found")
        print("Please execute the command")
        print("source download_dataset.sh")
        import sys
        sys.exit(-1)
    with open("data/person/ConfLongDemo_JSI.txt", "r") as f:
        current_person = "A01"

        for line in f:
            arr = line.split(",")
            if (len(arr) < 6):
                break
            if (arr[0] != current_person):
                # Enque and reset
                series_x = np.stack(series_x, axis=0)
                series_t = np.stack(series_t, axis=0)
                series_y = np.array(series_y, dtype=np.int32)
                all_x.append(series_x)
                all_t.append(series_t)
                all_y.append(series_y)
                last_millis = None
                series_x = []
                series_y = []
                series_t = []

            millis = np.int64(arr[2]) / (100 * 1000)
            # 100ms will be normalized to 1.0
            millis_mapped_to_1 = 10.0
            if (last_millis is None):
                elasped_sec = 0.05
            else:
                elasped_sec = float(millis - last_millis) / 1000.0
            elasped = elasped_sec * 1000 / millis_mapped_to_1

            last_millis = millis
            all_elapsed.append(elasped)
            current_person = arr[0]
            sensor_id = sensor_ids[arr[1]]
            label_col = class_map[arr[7].replace("\n", "")]
            feature_col_2 = np.array(arr[4:7], dtype=np.float32)
            # Last 3 entries of the feature vector contain sensor value

            # First 4 entries of the feature vector contain sensor ID
            feature_col_1 = np.zeros(4, dtype=np.float32)
            feature_col_1[sensor_id] = 1

            feature_col = np.concatenate([feature_col_1, feature_col_2])
            series_x.append(feature_col)
            series_t.append(elasped)
            all_feats.append(feature_col)
            series_y.append(label_col)

    all_feats = np.stack(all_feats, axis=0)
    print("all_feats.shape: ", str(all_feats.shape))
    all_elapsed = np.stack(all_elapsed, axis=0)
    print("all_elapsed.mean", str(np.mean(all_elapsed)))
    print("all_elapsed.med ", str(np.median(all_elapsed)))

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

    return all_x, all_t, all_y


def cut_in_sequences(all_x, all_t, all_y, seq_len, inc=1):
    sequences_x = []
    sequences_t = []
    sequences_y = []

    for i in range(len(all_x)):
        x, t, y = all_x[i], all_t[i], all_y[i]

        for s in range(0, x.shape[0] - seq_len, inc):
            start = s
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_t.append(t[start:end])
            sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=0), np.stack(sequences_t, axis=0).reshape([-1, seq_len, 1]), np.stack(sequences_y,
                                                                                                            axis=0)

class PersonData:

    def __init__(self, seq_len=32):
        self.seq_len = seq_len
        self.num_classes = 7
        all_x, all_t, all_y = load_crappy_formated_csv()
        all_x, all_t, all_y = cut_in_sequences(all_x, all_t, all_y, seq_len=seq_len, inc=seq_len // 2)

        print("all_x.shape: ", str(all_x.shape))
        print("all_t.shape: ", str(all_t.shape))
        print("all_y.shape: ", str(all_y.shape))
        total_seqs = all_x.shape[0]
        print("Total number of sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(98841).permutation(total_seqs)
        test_size = int(0.2 * total_seqs)

        self.test_x = all_x[permutation[:test_size]]
        self.test_y = all_y[permutation[:test_size]]
        self.test_t = all_t[permutation[:test_size]]
        self.train_x = all_x[permutation[test_size:]]
        self.train_t = all_t[permutation[test_size:]]
        self.train_y = all_y[permutation[test_size:]]

        self.feature_size = int(self.train_x.shape[-1])

        print("train_x.shape: ", str(self.train_x.shape))
        print("train_t.shape: ", str(self.train_t.shape))
        print("train_y.shape: ", str(self.train_y.shape))
        print("Total number of train sequences: {}".format(self.train_x.shape[0]))
        print("Total number of test  sequences: {}".format(self.test_x.shape[0]))


    def iterate_train(self,batch_size=1024):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_t = self.train_t[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            yield (batch_x,batch_t,batch_y)

class PersonModel:

    def __init__(self,model_type,model_size,sparsity_level=0.0):
        self.model_type = model_type
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,4+3])
        self.target_y = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None,None])

        self.model_size = model_size
        learning_rate = args.lr # LTC needs a higher learning rate

        if model_type == "solution_ltc" or model_type == "solution_node":
            self.t = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,1])
            head = tf.concat((self.x, self.t), axis=-1)
        else:
            head = self.x


        if(model_type == "lstm"):
            # unstacked_signal = tf.unstack(x,axis=0)
            self.fused_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            self.wm = ltc.cells.LTC(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.compat.v1.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())

        elif(model_type == "node"):
            self.fused_cell = ltc.cells.node.NODE(model_size,cell_clip=-1)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)

        elif (model_type == "solution_ltc"):
            self.fused_cell = ltc.cells.LTCSolution(model_size)
            head, _ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=False)

        elif (model_type == "solution_node"):
            self.fused_cell = ltc.cells.node_solution.NODESolution(model_size)
            head, _ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)

        elif(model_type == "ctgru"):
            self.fused_cell = ltc.cells.CTGRU(model_size,cell_clip=-1)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)

        elif(model_type == "ctrnn"):
            self.fused_cell = ltc.cells.CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)

        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())

        self.y = tf.compat.v1.layers.Dense(7,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(input_tensor=tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.compat.v1.InteractiveSession()
        # config = tf.ConfigProto()
        # config.intra_op_parallelism_threads = 4
        # config.inter_op_parallelism_threads = 1
        # self.sess = tf.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())

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

        self.saver = tf.compat.v1.train.Saver()

    def get_sparsity_ops(self):
        tf_vars = tf.compat.v1.trainable_variables()
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
        v_assign_op = tf.compat.v1.assign(v,v*mask)
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
                feed_test = {self.x:gesture_data.test_x,self.target_y: gesture_data.test_y}
                feed_val = {self.x:gesture_data.test_x,self.target_y: gesture_data.test_y}

                if self.model_type == "solution_ltc" or self.model_type == "solution_node":
                    feed_test[self.t] = gesture_data.test_t
                    feed_val[self.t] = gesture_data.test_t

                test_acc,test_loss = self.sess.run([self.accuracy,self.loss], feed_test)

                if(test_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = test_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        test_loss,test_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_t,batch_y in gesture_data.iterate_train(batch_size=args.batch_size):
                feed_train = {self.x: batch_x, self.target_y: batch_y}
                if self.model_type == "solution_ltc" or self.model_type == "solution_node":
                    feed_train[self.t] = batch_t
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step], feed_train)
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                elapsed = time.time()-start_time
                print("Epochs {:03d} ({:0.1f} s), train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,elapsed,
                    np.mean(losses),np.mean(accs)*100,
                    test_loss,test_acc*100,
                    test_loss,test_acc*100
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,test_loss,test_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            test_loss,test_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            test_loss,test_acc,
            test_loss,test_acc
        ))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="solution_ltc")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=64,type=int)
    parser.add_argument('--batch_size',default=1024,type=int)
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--epochs',default=1000,type=int)
    parser.add_argument('--sparsity',default=0,type=float)
    args = parser.parse_args()

    person_data = PersonData()
    person_data.train_t = (person_data.train_t - np.min(person_data.train_t)) / (np.max(person_data.train_t) - np.min(person_data.train_t))
    person_data.train_t = (person_data.test_t - np.min(person_data.test_t)) / (np.max(person_data.test_t) - np.min(person_data.test_t))

    model = PersonModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity)

    model.fit(person_data,epochs=args.epochs,log_period=args.log)
