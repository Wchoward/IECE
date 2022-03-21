import os, time, sys
# sys.path.append('./utils')
from utils.file_utils import *
from utils.prepare_data import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from model import MainModel
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from absl import app, flags, logging
from keras import backend as K

FLAGS = flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
flags.DEFINE_string('dataset_path', 'data/all_data.csv', 'emotion category dataset path')
flags.DEFINE_string('NRC_filepath', 'data/NRC.json', 'NRC_VAD json file path')
flags.DEFINE_string('conceptnet_dict_filepath', 'data/simplified_expand_concept_dict.json', 'ConceptNet dict file path')
flags.DEFINE_string('Dataset', 'implicit', 'explicit Dataset or implicit dataset')
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
flags.DEFINE_string('w2v_ori_file', 'data/w2v_origin_300.txt', 'origin embedding file')
flags.DEFINE_string('w2v_concept_file', 'data/filtered_expand_conceptnet_embedding.txt', 'concept embedding file')
flags.DEFINE_integer('embedding_dim', 300, 'dimention of word embedding')
flags.DEFINE_float('lam', 0.25, 'lambda for emotion intensity of VAD')
## input struct ##
flags.DEFINE_integer('max_doc_len', 169, 'max number of tokens per document')
flags.DEFINE_integer('max_event_len', 44, 'max number of tokens per event candidate')
## model struct ##
flags.DEFINE_integer('n_hidden', 64, 'number of hidden unit')
flags.DEFINE_integer('n_cause_class', 2, 'number of distinct cause class')
flags.DEFINE_integer('n_emotion_class', 7, 'number of distinct emotion class')
flags.DEFINE_string('ori_doc_module_type', 'attention', 'module type of origin doc: attention, lstm, cnn')
flags.DEFINE_string('candi_event_module_type', 'attention', 'module type of candidate event: attention, lstm, cnn')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
flags.DEFINE_string('log_file_name', '', 'name of log file')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('epoch', 8, 'epochs')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_boolean('integrate_ek', True, 'intergrate external knowledge: True, False')




def run():
    if FLAGS.log_file_name:
        if not os.path.exists('log'):
            os.makedirs('log')
        sys.stdout = open(FLAGS.log_file_name, 'w')
    print_time()
    print('############### Loading Data ###############')
    print('\nload NRC_VAD... ')
    NRC = load_dict_json(FLAGS.NRC_filepath)
    print('NRC_VAD words: {}'.format(len(NRC)))
    print('load NRC_VAD done!\n')
    
    print('\nload ConceptNet dict... ')
    concept_dict = load_dict_json(FLAGS.conceptnet_dict_filepath)
    print('ConceptNet dict lenth: {}'.format(len(concept_dict)))
    print('load concpetnet dict done!\n')

    print('\nload Dataset... ')
    X_train, X_test, y_train, y_test = load_split_dataset(FLAGS.dataset_path)
    print('Train data length: {}, Test data length: {}'.format(len(X_train), len(X_test)))
    # X_train, y_train = load_dataset('data/train_explicit_data.csv')
    # X_test, y_test = load_dataset('data/test_explicit_data.csv')
    print('load dataset done!\n')

    print('############### Loading Embedding ###############')
    print('\nload origin word2vector... ')
    origin_word2vec = KeyedVectors.load_word2vec_format(FLAGS.w2v_ori_file, binary=False)
    print('Origin word2vector size: {}'.format((len(origin_word2vec.wv.vocab), origin_word2vec.wv.vector_size)))
    print('load origin word2vector done!\n')

    print('\nload conceptNet word2vector... ')
    conceptnet_word2vec = KeyedVectors.load_word2vec_format(FLAGS.w2v_concept_file, binary=False)
    print('ConceptNet word2vector size: {}'.format((len(conceptnet_word2vec.wv.vocab), conceptnet_word2vec.wv.vector_size)))
    print('load conceptNet word2vector done!\n')

    print('############### Start Encoding ###############')
    train_emotion_labels = get_onehot_encoding(y_train[:, 0])
    test_emotion_labels = get_onehot_encoding(y_test[:, 0])
    train_cause_labels = get_onehot_encoding(y_train[:, 1])
    test_cause_labels = get_onehot_encoding(y_test[:, 1])

    train_original_text = get_sent_embedding_integrated_conceptnet(X_train[:,0], FLAGS.max_doc_len, concept_dict, FLAGS.lam, origin_word2vec, conceptnet_word2vec, NRC, FLAGS.integrate_ek)
    test_original_text = get_sent_embedding_integrated_conceptnet(X_test[:,0], FLAGS.max_doc_len, concept_dict, FLAGS.lam, origin_word2vec, conceptnet_word2vec, NRC, FLAGS.integrate_ek)
    train_event_text = get_sent_embedding_integrated_conceptnet(X_train[:,1], FLAGS.max_event_len, concept_dict, FLAGS.lam, origin_word2vec, conceptnet_word2vec, NRC, FLAGS.integrate_ek)
    test_event_text = get_sent_embedding_integrated_conceptnet(X_test[:,1], FLAGS.max_event_len, concept_dict, FLAGS.lam, origin_word2vec, conceptnet_word2vec, NRC, FLAGS.integrate_ek)

   

    with tf.name_scope("input_module"):
        original_text_input = layers.Input(batch_shape=(None, FLAGS.max_doc_len, FLAGS.embedding_dim))
        event_input = layers.Input(batch_shape=(None, FLAGS.max_event_len, FLAGS.embedding_dim))
        emotion_labels_input = layers.Input(batch_shape=(None, FLAGS.n_emotion_class))
        emotion_cause_labels_input = layers.Input(batch_shape=(None, 2))
        # predictions = layers.Input(batch_shape=(None, 1))

    net = MainModel()
    model = net.model_build(FLAGS.ori_doc_module_type, FLAGS.candi_event_module_type, original_text_input, event_input, emotion_labels_input, emotion_cause_labels_input)

    optimizer = tf.keras.optimizers.RMSprop(FLAGS.learning_rate)
    model.compile(optimizer=optimizer, 
        # loss={'out_cause':binary_focal_loss(),'out_emotion':'categorical_crossentropy'}, loss_weights={'out_cause': 0.8, 'out_emotion':10.},
        metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    print('train_original_text size: {}'.format(train_original_text.shape))
    print('train_event_text size: {}'.format(train_event_text.shape))
    print('train_emotion_labels size: {}'.format(train_emotion_labels.shape))
    print('train_cause_labels size: {}'.format(train_cause_labels.shape))

    print('test_original_text size: {}'.format(test_original_text.shape))
    print('test_event_text size: {}'.format(test_event_text.shape))
    print('test_emotion_labels size: {}'.format(test_emotion_labels.shape))
    print('test_cause_labels size: {}'.format(test_cause_labels.shape))

    model.fit(x=[train_original_text, train_event_text, train_emotion_labels, train_cause_labels], y=[train_emotion_labels, train_cause_labels], epochs=FLAGS.epoch,
              shuffle=True, validation_data=([test_original_text, test_event_text, test_emotion_labels, test_cause_labels], [test_emotion_labels, test_cause_labels]),
              batch_size=16, callbacks=[early_stopping, reduce_lr])

    cur_time = time.localtime(time.time())
    # model.save(os.path.join('model/my_model',
    #                         'my_model_{}_{}_{}_{}_{}.h5'.format(cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour,
    #                                                             cur_time.tm_min, cur_time.tm_sec)))

    # test
    prediction = model.predict([test_original_text, test_event_text, test_emotion_labels, test_cause_labels])

    p, r, f1 = cal_prf(prediction, test_emotion_labels, 'emotion')
    print('emotion_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))

    p, r, f1 = cal_prf(prediction, test_cause_labels, 'cause')
    print('cause_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))



def main(_):
    module = ['cnn', 'lstm', 'attention']
    # FLAGS.log_file_name = 'log/doc({})_candi({})_lam({}).log'.format(FLAGS.ori_doc_module_type, FLAGS.candi_event_module_type, FLAGS.lam)
    # FLAGS.log_file_name = 'log/explicit_dataset.log'
    run()
    # for FLAGS.ori_doc_module_type in module:
    #     for FLAGS.candi_event_module_type in module:
    #         run()


if __name__ == "__main__":
    app.run(main)