from inits import *
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def sparse_dense_matmul_batch(sp_a, b):

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)


def dot(x, y, sparse=False):
    """Wrapper for 3D tf.matmul (sparse vs dense)."""
    if sparse:
        res = sparse_dense_matmul_batch(x, y)
    else:
        res = tf.einsum('bij,jk->bik', x, y) # tf.matmul(x, y)
    return res


def gru_unit(FFD_info,TRT_info,FFD_weight,TRT_weight,support, x, var, act, mask, dropout, sparse_inputs=False):
    """GRU unit with 3D tensor inputs."""
    # message passing
    print('gru_inputx_shape:', x.shape)
    print('gru_support_shape:', x.shape)
    support = tf.nn.dropout(support, dropout) # optional
    print(x.shape,support.shape,FFD_info.shape)
    # support = support * FFD_weight
    a = tf.matmul(support, x)
    # a = dot(a,var['weights_a'],sparse_inputs)+var['bias_a']
    # exit(0)
    # a=tf.add(a,FFD_weight)
    a = a*(1+FFD_weight)
    # update gate        
    z0 = dot(a, var['weights_z0'], sparse_inputs) + var['bias_z0']
    z1 = dot(x, var['weights_z1'], sparse_inputs) + var['bias_z1'] 
    z = tf.sigmoid(z0 + z1)
    
    # reset gate
    r0 = dot(a, var['weights_r0'], sparse_inputs) + var['bias_r0']
    r1 = dot(x, var['weights_r1'], sparse_inputs) + var['bias_r1']
    r = tf.sigmoid(r0 + r1)

    # update embeddings    
    h0 = dot(a, var['weights_h0'], sparse_inputs) + var['bias_h0']
    h1 = dot(r*x, var['weights_h1'], sparse_inputs) + var['bias_h1']
    h = act(mask * (h0 + h1))
    h = act((h0 + h1))

    return h*z + x*(1-z)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        print('dense_input_shape:', x.shape)
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']
        print('dense_input_shape:', x.shape)
        return self.act(output)


class GraphLayer(Layer):
    """Graph layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, steps=2, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mask = placeholders['mask']
        self.steps = steps

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.FFD_weight=placeholders['FFD_weight']
        self.TRT_weight = placeholders['TRT_weight']
        self.FFD_info = placeholders['FFD_info']
        self.TRT_info = placeholders['TRT_info']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_encode'] = glorot([input_dim, output_dim],
                                                    name='weights_encode')
            self.vars['weights_a'] = glorot([output_dim, output_dim], name='weights_a')
            self.vars['weights_z0'] = glorot([output_dim, output_dim], name='weights_z0')
            self.vars['weights_z1'] = glorot([output_dim, output_dim], name='weights_z1')
            self.vars['weights_r0'] = glorot([output_dim, output_dim], name='weights_r0')
            self.vars['weights_r1'] = glorot([output_dim, output_dim], name='weights_r1')
            self.vars['weights_h0'] = glorot([output_dim, output_dim], name='weights_h0')
            self.vars['weights_h1'] = glorot([output_dim, output_dim], name='weights_h1')

            self.vars['bias_encode'] = zeros([output_dim], name='bias_encode')
            self.vars['bias_a'] = zeros([output_dim], name='bias_a')
            self.vars['bias_z0'] = zeros([output_dim], name='bias_z0')
            self.vars['bias_z1'] = zeros([output_dim], name='bias_z1')
            self.vars['bias_r0'] = zeros([output_dim], name='bias_r0')
            self.vars['bias_r1'] = zeros([output_dim], name='bias_r1')
            self.vars['bias_h0'] = zeros([output_dim], name='bias_h0')
            self.vars['bias_h1'] = zeros([output_dim], name='bias_h1')
            

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        print('gnn_input_shape:',x.shape)
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # encode inputs
        x = dot(x, self.vars['weights_encode'], 
                self.sparse_inputs) + self.vars['bias_encode']
        # output = self.mask * (self.act(x)+self.weight)
        output = self.mask * self.act(x)
        output = self.act(x)
        # convolve
        for _ in range(self.steps):
            output = gru_unit(self.FFD_info,self.TRT_info,self.FFD_weight,self.TRT_weight,self.support, output, self.vars, self.act,
                              self.mask, 1-self.dropout, self.sparse_inputs)

        print('gnn_output_shape:', x.shape)
        return output

class ReadoutLayer(Layer):
    """Graph Readout Layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.mask = placeholders['mask']
        self.FFD_weight = placeholders['FFD_weight']
        self.TRT_weight = placeholders['TRT_weight']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_att'] = glorot([input_dim, 1], name='weights_att')
            self.vars['weights_emb'] = glorot([input_dim, input_dim], name='weights_emb')
            self.vars['weights_mlp'] = glorot([input_dim, output_dim], name='weights_mlp')

            self.vars['bias_att'] = zeros([1], name='bias_att')
            self.vars['bias_emb'] = zeros([input_dim], name='bias_emb')
            self.vars['bias_mlp'] = zeros([output_dim], name='bias_mlp')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # soft attention
        att = tf.sigmoid(dot(x, self.vars['weights_att']) + self.vars['bias_att'])
        # att = tf.add(att,self.TRT_weight)

        emb = self.act(dot(x, self.vars['weights_emb']) + self.vars['bias_emb'])

        N = tf.reduce_sum(self.mask, axis=1)
        M = (self.mask-1) * 1e9
        
        # graph summation
        h=att * emb
        h_emb = self.mask * att * emb # self-attention
        # h_plus=self.mask * (att+self.weight) * emb
        # h_plus = (att + self.TRT_weight+self.FFD_weight) * emb
        h_plus = (att + self.TRT_weight ) * emb
        # h_plus = (att +  self.FFD_weight) * emb
        print('att.shape:',att.shape)
        print('emb_shape:',emb.shape)
        print('h_emb_shape:', h_emb.shape)
        # h_eye = x * eye-based attention
        print('self.TRT_weight:',self.TRT_weight.shape)
        h_eye = self.mask * self.TRT_weight * emb
        print('h_eye_shape:',h_eye.shape)
        print('h_eye:',type(h_eye))

        # h_concat=np.concatenate([h_emb,h_eye],axis=1)
        # h_concat = tf.concat([h_emb,h_eye],axis=1,name='concat')
        # h_concat = h_emb+h_eye
        # print('h_concat_shape:', self.act(h_concat))
        
        
        # g = tf.reduce_sum(h_eye, axis=1) / N + tf.reduce_max(h_eye + M, axis=1)
        # g = tf.reduce_sum(h, axis=1) / N + tf.reduce_max(h + M, axis=1)
        g=tf.reduce_sum(h, axis=1) / N + tf.reduce_max(h+M , axis=1)
        # g = tf.reduce_sum(h_1, axis=1) / N + tf.reduce_max(h_1 + M, axis=1)
        # g = tf.reduce_sum(h_plus, axis=1) / N + tf.reduce_max(h_plus + M, axis=1)
        g=tf.reduce_max(h_plus+M, axis=1)
        g = tf.reduce_sum(h_plus, axis=1) / N
        # g = tf.reduce_sum(h_plus, axis=1) / N + tf.reduce_max(h + M, axis=1)
        # mr_emb_concat_eye
        # g = tf.reduce_sum(h, axis=1)/N+tf.reduce_max(h_plus, axis=1)

        # mr_concat_g
        # g=h_concat
        # g=tf.reduce_max(h_concat +M, axis=1)

        # mr_sum_concat
        # g = tf.reduce_sum(h_concat, axis=1) / N

        # mr_emb_eye_weight
        # g = self.mask * att * emb + self.weight
        # g=h_plus
        # g = tf.reduce_sum(g, axis=1) / N + tf.reduce_max(g + M, axis=1)
        g = tf.nn.dropout(g, 1-self.dropout)


        # classification
        output = tf.matmul(g, self.vars['weights_mlp']) + self.vars['bias_mlp']

        return output

