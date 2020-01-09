import numpy as np
import tensorflow as tf
import data_load

DATA_PATH = "data"
HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35
DECAY_RATE = 0.98

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP =1
NUM_EPOCH = 6
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

class LModel:
    def __init__(self,is_training,batch_size,num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lr = tf.Variable(0.0, trainable=False)

        self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob = KEEP_PROB)
        lm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS)

        self.initial_state = lm_cell.zero_state(batch_size,tf.float32)
        embedding = tf.get_variable("embedding",[VOCAB_SIZE,HIDDEN_SIZE])
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)

        if is_training and KEEP_PROB < 1:
            inputs = tf.nn.dropout(inputs,KEEP_PROB)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,state = lm_cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)
        output = tf.concat(outputs,1)
        output = tf.reshape(output,[-1,HIDDEN_SIZE])

        weight = tf.get_variable("weight",[HIDDEN_SIZE,VOCAB_SIZE])
        bias = tf.get_variable("bias",[VOCAB_SIZE])
        logits = tf.matmul(output,weight)+bias

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(self.targets,[-1])],[tf.ones([batch_size*num_steps],dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss)/batch_size
        self.final_state = state

        if not is_training: return
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost,tvars),MAX_GRAD_NORM)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


def run_epoch(session,model,data,train_op,output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step,(x,y) in enumerate(data_load.data_iterator(data,model.batch_size,model.num_steps)):
        cost,state,_=session.run([model.cost,model.final_state,train_op],{model.input_data:x,model.targets:y,model.initial_state:state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step%100 == 0:
            print("After %d steps,perplexity is %.3f"%(step,np.exp(total_costs/iters)))


    return np.exp(total_costs/iters)
'''
def main(self):
    train_data = data_load.load_train()
    test_data = data_load.load_test()
    valid_data = data_load.load_valid()

    initializer = tf.random_uniform_initializer(-0.05,0.05)
    with tf.variable_scope("language_model",reuse=None,initializer=initializer):
        train_model = LModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)

    with tf.variable_scope("language_model",reuse=True,initializer=initializer):
        eval_model = LModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for i in range(NUM_EPOCH):
            print("In iteration: %d" %(i+1))
            run_epoch(session,train_model,train_data,train_model.train_op,True)
            valid_perplexity = run_epoch(session,eval_model,valid_data,tf.no_op(),False)
            print("Epoch:%d Validation Perplexity: %.3f" %(i+1,valid_perplexity))

        test_perplexity = run_epoch(session,eval_model,test_data,tf.no_op(),False)
        print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == '__main__':
    tf.app.run()
'''