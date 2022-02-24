import os
import json
from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle


def get_emotion_intensity(NRC, word, lam):
    """
    get emotion intensity from NRC_VAD
    Args:
        NRC: NRC dict
        word: target word
        lam: lambda
    Returns:
        intensity value
    """
    if word not in NRC:
        return None
    v, a, d = NRC[word]
    w = lam * v + (1 - lam) * a
    return w


def load_dataset(input_file):
    df = pd.read_csv(input_file)
    df = shuffle(df)
    X = df.iloc[:, [0, 1]].values
    y = df.iloc[:, [2, 3]].values
    return X, y


def load_split_dataset(input_file):
    """
    load all_data.csv
    """
    X, y = load_dataset(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, stratify=y[:,0])
    return X_train, X_test, y_train, y_test


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx, :]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], axis=0)
            y_train = np.concatenate([y_train, y_part], axis=0)
    return X_train, y_train, X_valid, y_valid


def get_onehot_encoding(lst):
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    le_lst = le.fit_transform(lst)
    return ohe.fit_transform(le_lst.reshape(-1, 1))
    

def get_sent_embedding_integrated_conceptnet(sent_lst, max_len, concept_dict, lam, origin_w2v_model, concpet_w2v_model, NRC, intergrated):
    seg_lst = []
    for sentence in sent_lst:
        seg_lst.append(str(sentence).split())
    sent_embedding_lst = []
    if intergrated == True:
        for sentence in seg_lst:
            word_embedding_lst = []
            for word in sentence:
                if word not in concept_dict.keys():
                    try:
                        word_embedding_lst.append(origin_w2v_model[word][np.newaxis,:])
                    except:
                        word_embedding_lst.append(origin_w2v_model['#'][np.newaxis,:])
                else:
                    weight_lst = []
                    concept_embedding_lst = []
                    for c_k in concept_dict[word]:
                        e_k = get_emotion_intensity(NRC, word, lam)
                        if e_k is not None:
                            if c_k['Concept'] in concpet_w2v_model:
                                weight_k = e_k * c_k['Weight']
                                weight_lst.append(weight_k)
                                concept_embedding_lst.append(concpet_w2v_model[c_k['Concept']])
                    weight_lst = np.array(weight_lst)
                    if not concept_embedding_lst:
                        integrated_embedding = origin_w2v_model[word]
                    else:
                        concept_embedding_lst = np.array(concept_embedding_lst)
                        alpha_lst = np.exp(weight_lst) / sum(np.exp(weight_lst))
                        integrated_embedding = (np.dot(alpha_lst, concept_embedding_lst) + origin_w2v_model[word]) / 2
                    word_embedding_lst.append(integrated_embedding[np.newaxis, :])
            sent_embedding = np.concatenate(word_embedding_lst[:max_len])
            sent_embedding_lst.append(
                np.pad(sent_embedding, ((0, max_len - len(sent_embedding)), (0, 0)), 'constant')[np.newaxis, :])
    if intergrated == False:
        #不引入外部知识
        for sentence in seg_lst:
            word_embedding_lst = []
            for word in sentence:
                try:
                    word_embedding_lst.append(origin_w2v_model[word][np.newaxis,:])
                except:
                    word_embedding_lst.append(origin_w2v_model['#'][np.newaxis,:])
            sent_embedding = np.concatenate(word_embedding_lst[:max_len])
            sent_embedding_lst.append(
                np.pad(sent_embedding, ((0, max_len - len(sent_embedding)), (0, 0)), 'constant')[np.newaxis, :])

    embedding = np.concatenate(sent_embedding_lst)
    return embedding


def cal_prf(pred_y, true_y, task):
    true_y = np.argmax(true_y, axis=1)
    if task == 'cause' :
        pred_y = np.argmax(pred_y[1], axis=1)
        p, r, f, _ = precision_recall_fscore_support(y_true=true_y, y_pred=pred_y, average='binary')
    if task == 'emotion' :
        pred_y = np.argmax(pred_y[0], axis=1)
        p, r, f, _ = precision_recall_fscore_support(y_true=true_y, y_pred=pred_y, average='macro')
        # p,r,f = [round(x, 3) for x in prf[:3]]
    return p, r, f

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed
