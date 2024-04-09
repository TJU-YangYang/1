import numpy as np

import tensorflow as tf

from time import time

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score

import os

import config

class PNN(BaseEstimator, TransformerMixin):

    def __init__(self, cate_feature_size, field_size, numeric_feature_size,

                 embedding_size=8, dropout_pnn=[1.0, 1.0],

                 deep_layers=[32, 32], deep_init_size = 100, dropout_deep=[0.5, 0.5, 0.5],

                 deep_layer_activation=tf.nn.relu,

                 epoch=10, batch_size=256,

                 learning_rate=0.001, optimizer="adam",

                 batch_norm=0, batch_norm_decay=0.995,

                 verbose=False, random_seed=2016,

                 use_pnn=True, use_deep=True,

                 loss_type="mse", eval_metric=roc_auc_score,

                 l2_reg=0.0, greater_is_better=True):

        assert (use_pnn or use_deep)

        assert loss_type in ["logloss", "mse"], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        print("cate_feature_size:")
        print(cate_feature_size)

        print("numeric_feature_size:")
        print(numeric_feature_size)

        print("field_size:")
        print(field_size)

        self.cate_feature_size = cate_feature_size

        self.numeric_feature_size = numeric_feature_size

        self.field_size = field_size

        self.embedding_size = embedding_size

        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size

        self.dropout_pnn = dropout_pnn

        self.deep_layers = deep_layers

        self.deep_init_size = deep_init_size

        self.num_pairs = int(self.deep_init_size * (self.deep_init_size - 1) / 2)

        self.dropout_dep = dropout_deep

        self.deep_layers_activation = deep_layer_activation

        self.use_pnn = use_pnn

        self.use_deep = use_deep

        self.l2_reg = l2_reg

        self.epoch = epoch

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.optimizer_type = optimizer


        self.batch_norm = batch_norm

        self.batch_norm_decay = batch_norm_decay



        self.verbose = verbose

        self.random_seed = random_seed

        self.loss_type = loss_type

        self.eval_metric = eval_metric

        self.greater_is_better = greater_is_better

        self.train_result,self.valid_result = [],[]

        tf.set_random_seed(self.random_seed)

        np.random.seed(self.random_seed)

        self._init_graph()



    def _init_graph(self):

        self.graph = tf.Graph()

        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            np.random.seed(self.random_seed)



            self.feat_index = tf.placeholder(tf.int32,

                                             shape=[None,None],

                                             name='feat_index')

            self.feat_value = tf.placeholder(tf.float32,

                                           shape=[None,None],

                                           name='feat_value')


            self.numeric_value = tf.placeholder(tf.float32,[None,None],name='num_value')

            self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')

            self.dropout_keep_pnn = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_pnn')

            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')

            self.train_phase = tf.placeholder(tf.bool,name='train_phase')



            self.weights = self._initialize_weights()



            # model

            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * 881 * 8

            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1]) # N * 881 *1

            self.embeddings = tf.multiply(self.embeddings,feat_value)  # N * 881 * 8

            print(self.embeddings.shape)


            self.x0 = tf.concat([self.numeric_value,

                                 tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])]

                                ,axis=1)# N * F

            print(self.x0.shape) # N * 26036

            self.x0 = tf.reshape(self.x0, shape=[-1, self.total_size])

            print(self.x0.shape) # N * 26036


            self.x0_bn = tf.layers.batch_normalization(self.x0, training = self.train_phase)


            if self.use_pnn:

                self.y_deep2= self.x0_bn

                # self.y_deep = self.x0_bn

                self.y_deep2 = tf.nn.dropout(self.y_deep2, self.dropout_keep_deep[0])


                for i in range(0,len(self.deep_layers)):

                    self.y_deep2 = tf.add(tf.matmul(self.y_deep2,self.weights["layer2_%d" %i]), self.weights["bias2_%d"%i])

                    self.y_deep2 = self.deep_layers_activation(self.y_deep2)

                    self.y_deep2 = tf.nn.dropout(self.y_deep2,self.dropout_keep_deep[i+1])


            if self.use_deep:

                print(self.use_pnn, self.use_deep)

                # # #
                # # # 无deep_init_size版本 # Quardatic Singal
                # # #
                # quadratic_output = []
                #
                #
                #
                # for i in range(self.deep_init_size):
                #
                #     theta = tf.multiply(self.x0_bn,tf.reshape(self.weights['product-quadratic-inner'][i],(1,-1))) # N * 26036(total_size)
                #
                #     quadratic_output.append(tf.reshape(tf.reduce_sum(theta,axis=1),shape=(-1,1))) # N * 1

                quadratic_output=tf.add(tf.matmul(self.x0_bn,tf.reshape(self.weights['product-quadratic-inner'] ,(self.total_size,-1))),self.weights['product-bias']) #N*init_deep_size

                self.lp = self.deep_layers_activation(quadratic_output)

                self.lp = tf.nn.dropout(quadratic_output,self.dropout_keep_deep[0])


                # self.lp = tf.concat(quadratic_output,axis=1) # N * init_deep_size



                self.num_pairs = int(self.deep_init_size * (self.deep_init_size - 1) / 2)


                row = []

                col = []

                for i in range(self.deep_init_size-1):

                    for j in range(i+1, self.deep_init_size):

                        row.append(i)

                        col.append(j)

                # batch * pair * k

                p = tf.transpose(

                    # pair * batch * k

                    tf.gather(

                        # num * batch * k

                        tf.transpose(

                            self.lp, [1, 0]),

                        row),

                    [1, 0])

                # batch * pair * k

                q = tf.transpose(

                    tf.gather(

                        tf.transpose(

                            self.lp, [1, 0]),

                        col),

                    [1, 0])

                p = tf.reshape(p, [-1, self.num_pairs])

                q = tf.reshape(q, [-1, self.num_pairs])

                self.ip = tf.reshape(p * q, [-1, self.num_pairs])

                self.y_deep= self.ip

                self.y_deep = tf.layers.batch_normalization(self.y_deep, training = self.train_phase)

                # self.y_deep = self.x0_bn

                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])


                for i in range(0,len(self.deep_layers)):

                    self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["layer_%d" %i]), self.weights["bias_%d"%i])

                    self.y_deep = self.deep_layers_activation(self.y_deep)

                    self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])


            #----DNNPNN---------

            if self.use_pnn and self.use_deep:

                concat_input = tf.concat([self.y_deep2, self.y_deep], axis=1)

                concat_input = tf.layers.batch_normalization(concat_input, training = self.train_phase)

            elif self.use_pnn:

                concat_input = tf.concat([self.y_deep2], axis=1)

            elif self.use_deep:

                concat_input = self.y_deep



            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])



            # loss

            if self.loss_type == "logloss":

                self.out = tf.nn.sigmoid(self.out)

                self.loss = tf.losses.log_loss(self.label, self.out)

            elif self.loss_type == "mse":

                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 regularization on weights

            if self.l2_reg > 0:

                self.loss += tf.contrib.layers.l2_regularizer(

                    self.l2_reg)(self.weights["concat_projection"])

                if self.use_deep:

                    for i in range(len(self.deep_layers)):

                        self.loss += tf.contrib.layers.l2_regularizer(

                            self.l2_reg)(self.weights["layer_%d" % i])

            if self.optimizer_type == "adam":

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,

                                                        epsilon=1e-8)

            elif self.optimizer_type == "adagrad":

                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,

                                                           initial_accumulator_value=1e-8)

            elif self.optimizer_type == "gd":

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            elif self.optimizer_type == "momentum":

                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):

    	        self.train_op = self.optimizer.minimize(self.loss)


            #init

            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()

            self.sess = tf.Session()

            self.sess.run(init)


            # number of params

            total_parameters = 0

            for variable in self.weights.values():

                shape = variable.get_shape()

                variable_parameters = 1

                for dim in shape:

                    variable_parameters *= dim.value

                total_parameters += variable_parameters

            if self.verbose > 0:

                print("#params: %d" % total_parameters)


    def _initialize_weights(self):

        tf.set_random_seed(self.random_seed)

        np.random.seed(self.random_seed)

        weights = dict()



        #embeddings


        weights['feature_embeddings'] = tf.Variable(

                tf.random_uniform([self.cate_feature_size,self.embedding_size]),

                name='feature_embeddings')
        if self.use_pnn:

            #deep layers
            np.random.seed(self.random_seed)

            num_layer = len(self.deep_layers)

            input_size = self.total_size

            # input_size = self.total_size

            glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))



            weights['layer2_0'] = tf.Variable(

                np.random.normal(loc=0,scale=glorot,size=(input_size,self.deep_layers[0])),dtype=np.float32, name='layer_0'

            )

            weights['bias2_0'] = tf.Variable(

                np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32, name='bias_0'

            )

            for i in range(1,num_layer):

                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))

                weights["layer2_%d" % i] = tf.Variable(

                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),

                    dtype=np.float32)  # layers[i-1] * layers[i]

                weights["bias2_%d" % i] = tf.Variable(

                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),

                    dtype=np.float32)  # 1 * layer[i]


        if self.use_deep:
        #Product Layers

            # 无deep_init版本
            weights['product-quadratic-inner'] = tf.Variable(tf.random_normal([self.deep_init_size,self.total_size],0.0,0.00001), name='product-quadratic-inner')


            weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size,],0.0,1.0), name='product-bias')


            #deep layers
            np.random.seed(self.random_seed)

            num_layer = len(self.deep_layers)

            input_size = self.num_pairs

            # input_size = self.total_size

            glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))


            weights['layer_0'] = tf.Variable(

                np.random.normal(loc=0,scale=glorot,size=(input_size,self.deep_layers[0])),dtype=np.float32, name='layer_0'

            )

            weights['bias_0'] = tf.Variable(

                np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32, name='bias_0'

            )

            for i in range(1,num_layer):

                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))

                weights["layer_%d" % i] = tf.Variable(

                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),

                    dtype=np.float32)  # layers[i-1] * layers[i]

                weights["bias_%d" % i] = tf.Variable(

                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),

                    dtype=np.float32)  # 1 * layer[i]


        # final concat projection layer



        if self.use_pnn and self.use_deep:

            input_size = self.deep_layers[-1] + self.deep_layers[-1]

        elif self.use_pnn:

            input_size = self.deep_layers[-1]

        elif self.use_deep:

            input_size = self.deep_layers[-1]


        glorot = np.sqrt(2.0/(input_size + 1))

        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32, name='concat_projection')

        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)


        return weights


    def get_batch(self,Xi,Xv,Xv2,y,batch_size,index):

        start = index * batch_size

        end = (index + 1) * batch_size

        end = end if end < len(y) else len(y)

        # print("get_batch:")
        #
        # print([[y_] for y_ in y[start:end]])

        return Xi[start:end],Xv[start:end],Xv2[start:end],[[y_] for y_ in y[start:end]]



    # shuffle three lists simutaneously

    def shuffle_in_unison_scary(self, a, b, c,d):

        np.random.seed(self.random_seed)

        rng_state = np.random.get_state()

        np.random.shuffle(a)

        np.random.set_state(rng_state)

        np.random.shuffle(b)

        np.random.set_state(rng_state)

        np.random.shuffle(c)

        np.random.set_state(rng_state)

        np.random.shuffle(d)


    def evaluate(self, Xi, Xv, Xv2, y):

        """

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param y: label of each sample in the dataset

        :return: metric of the evaluation

        """

        y_pred = self.predict(Xi, Xv, Xv2)

        return self.eval_metric(y, y_pred)


    def predict(self, Xi, Xv, Xv2):

        """

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :return: predicted probability of each sample

        """

        # dummy y

        dummy_y = [1] * len(Xi)

        batch_index = 0

        Xi_batch, Xv_batch, Xv2_batch, y_batch = self.get_batch(Xi, Xv, Xv2, dummy_y, self.batch_size, batch_index)

        y_pred = None

        while len(Xi_batch) > 0:

            num_batch = len(y_batch)

            feed_dict = {self.feat_index: Xi_batch,

                         self.feat_value: Xv_batch,

                         self.label: y_batch,

                         self.numeric_value: Xv2_batch,

                         self.dropout_keep_pnn: self.dropout_pnn,

                        self.dropout_keep_deep: [1.0] * len(self.dropout_dep),

                         self.train_phase: False}

            batch_out = self.sess.run(self.out, feed_dict=feed_dict)



            if batch_index == 0:

                y_pred = np.reshape(batch_out, (num_batch,))

            else:

                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1

            Xi_batch, Xv_batch,Xv2_batch, y_batch = self.get_batch(Xi, Xv, Xv2, dummy_y, self.batch_size, batch_index)


        return y_pred


    def fit_on_batch(self,Xi,Xv,Xv2,y):

        feed_dict = {self.feat_index:Xi,

                     self.feat_value:Xv,

                     self.numeric_value:Xv2,

                     self.label:y,

                     self.dropout_keep_pnn:self.dropout_pnn,

                     self.dropout_keep_deep:self.dropout_dep,

                     self.train_phase:True}



        loss,train_op,label,out = self.sess.run([self.loss,self.train_op,self.label, self.out], feed_dict=feed_dict)
        # print("fit:")
        # print(loss)
        # print(label)
        # print(out)
        # print(weights)
        # print(x0)
        # print(x0_bn)

        return loss


    def fit(self,  cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train,

            cate_Xi_valid=None, cate_Xv_valid=None, numeric_Xv_valid=None,y_valid=None,

            early_stopping=False, refit=False):

        """

        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]

                         indi_j is the feature index of feature field j of sample i in the training set

        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]

                         vali_j is the feature value of feature field j of sample i in the training set

                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)

        :param y_train: label of each sample in the training set

        :param Xi_valid: list of list of feature indices of each sample in the validation set

        :param Xv_valid: list of list of feature values of each sample in the validation set

        :param y_valid: label of each sample in the validation set

        :param early_stopping: perform early stopping or not

        :param refit: refit the model on the train+valid dataset or not

        :return: None

        """

        has_valid = cate_Xv_valid is not None

        for epoch in range(self.epoch):

            t1 = time()

            self.shuffle_in_unison_scary(cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train)

            total_batch = int(len(y_train) / self.batch_size)

            for i in range(total_batch):

                cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train, self.batch_size, i)

                ##loss
                self.fit_on_batch(cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch)


            # evaluate training and validation datasets

            train_result = self.evaluate(cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train)

            self.train_result.append(train_result)

            if has_valid:

                valid_result = self.evaluate(cate_Xi_valid, cate_Xv_valid,numeric_Xv_valid, y_valid)

                self.valid_result.append(valid_result)

            if self.verbose > 0 and epoch % self.verbose == 0:

                if has_valid:

                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"

                        % (epoch + 1, train_result, valid_result, time() - t1))

                else:

                    print("[%d] train-result=%.4f [%.1f s]"

                        % (epoch + 1, train_result, time() - t1))

            if has_valid and early_stopping and self.training_termination(self.valid_result):

                break



        # fit a few more epoch on train+valid until result reaches the best_train_score

        if has_valid and refit:

            if self.greater_is_better:

                best_valid_score = max(self.valid_result)

            else:

                best_valid_score = min(self.valid_result)

            best_epoch = self.valid_result.index(best_valid_score)

            best_train_score = self.train_result[best_epoch]

            cate_Xi_train = cate_Xi_train + cate_Xi_valid

            cate_Xv_train = cate_Xv_train + cate_Xv_valid

            numeric_Xv_train = numeric_Xv_train + numeric_Xv_train

            y_train = y_train + y_valid

            for epoch in range(100):

                self.shuffle_in_unison_scary(cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train)

                total_batch = int(len(y_train) / self.batch_size)

                for i in range(total_batch):

                    cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train, self.batch_size, i)

                    self.fit_on_batch(cate_Xi_batch, cate_Xv_batch, numeric_Xv_batch, y_batch)

                # check

                train_result = self.evaluate(cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train)

                if abs(train_result - best_train_score) < 0.001 or (self.greater_is_better and train_result > best_train_score) or ((not self.greater_is_better) and train_result < best_train_score):

                    break

    def training_termination(self, valid_result):

        if len(valid_result) > 5:

            if self.greater_is_better:

                if valid_result[-1] < valid_result[-2] and valid_result[-2] < valid_result[-3] and valid_result[-3] < valid_result[-4] and valid_result[-4] < valid_result[-5]:

                    return True

            else:

                if valid_result[-1] > valid_result[-2] and valid_result[-2] > valid_result[-3] and valid_result[-3] > valid_result[-4] and valid_result[-4] > valid_result[-5]:

                    return True

        return False


    def save_result(self, i):

        filename = "Model/"+str(i)+"model.ckpt"

        self.saver.save(self.sess, os.path.join(config.SUB_DIR, filename))

