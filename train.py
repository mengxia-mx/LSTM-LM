import time
import os
import codecs
import _pickle as cPickle
import data_load
import model
import tensorflow as tf

def train():
    start = time.time()
    save_dir = "model"
    output = "nohup.out"
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)


    train_data = data_load.load_train()
    valid_data = data_load.load_valid()

    out_file = os.path.join(save_dir, output)
    fout = codecs.open(out_file, "w", encoding="UTF-8")


    print("Begin training...")


    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        # Build models
        with tf.variable_scope("language_model", reuse=None, initializer=initializer):
            train_model = model.LModel(True,model.TRAIN_BATCH_SIZE,model.TRAIN_NUM_STEP)
        with tf.variable_scope("language_model", reuse=True, initializer=initializer):
            eval_model = model.LModel(True,model.TRAIN_BATCH_SIZE,model.TRAIN_NUM_STEP)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        tf.initialize_all_variables().run()
        dev_pp = 10000000.0
        # process each epoch
        e = 0
        decay_counter = 1
        learning_rate = model.LEARNING_RATE
        while e < model.NUM_EPOCH:
            if e > 4:
                lr_decay = model.DECAY_RATE ** decay_counter
                learning_rate = model.LEARNING_RATE * lr_decay
                decay_counter += 1
            print("Epoch: %d" % (e + 1))
            train_model.assign_lr(sess, learning_rate)
            print("Learning rate: %.3f" % sess.run(train_model.lr))

            train_perplexity = model.run_epoch(sess,train_model,train_data,train_model.train_op,True)
            print("Train Perplexity: %.3f" % train_perplexity)

            dev_perplexity = model.run_epoch(sess,eval_model,valid_data,tf.no_op(),False)
            print("Valid Perplexity: %.3f" % dev_perplexity)

            # write results to file
            fout.write("Epoch: %d\n" % (e + 1))
            fout.write("Learning rate: %.3f\n" % sess.run(train_model.lr))
            fout.write("Train Perplexity: %.3f\n" % train_perplexity)
            fout.write("Valid Perplexity: %.3f\n" % dev_perplexity)
            fout.flush()

            if dev_pp > dev_perplexity:
                print ("Achieve highest perplexity on dev set, save model.")
                checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print ("model saved to {}".format(checkpoint_path))
                dev_pp = dev_perplexity
            e += 1

        print("Training time: %.0f" % (time.time() - start))
        fout.write("Training time: %.0f\n" % (time.time() - start))

def main():
    train()

if __name__ == '__main__':
    main()