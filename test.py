import time
import _pickle as cPickle
import data_load
import model
import tensorflow as tf

def test():
    save_dir = "./model"
    start = time.time()
    test_data = data_load.load_test()
    print("Begin testing...")
    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            testmodel = model.LModel(False,model.EVAL_BATCH_SIZE,model.EVAL_NUM_STEP)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        test_perplexity = model.run_epoch(sess,testmodel,test_data,tf.no_op(),False)
        print("Test Perplexity: %.3f" % test_perplexity)
        print("Test time: %.0f" % (time.time() - start))

def main():
    test()

if __name__ == '__main__':
    main()