from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional,Dropout, SimpleRNN, GRU, LSTM)


def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Called at the end of the final_model.
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def final_model(input_dim, filters, kernel_size, conv_stride,
        conv_border_mode, units, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add bidirectional recurrent layer
    #rnn=LSTM(units, activation=activation,return_sequences=True, implementation=2)
    rnn = GRU(units, activation="relu", return_sequences=True, implementation=2)
    bidir_rnn = Bidirectional(rnn, name="bidir_rnn")(bn_cnn)
    # Add batch normalization 
    bn_rnn = BatchNormalization(name='batchnorm_rnn')(bidir_rnn)
    # Dropout layer
    dropout = Dropout(0.3, name='dropout')(bn_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(dropout)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model