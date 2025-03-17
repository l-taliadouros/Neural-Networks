'''
E4040 2024Fall Assignment3
Machine Translation
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras import Sequential


class Encoder(tf.keras.layers.Layer):
    '''
    The encoder section of our NMT model. 
    
    It will consist of:
    1. An embedding layer to project the input language words (tokens) to hidden_size
    2. An RNN layer to encode the sequence (in this case, we use the LSTM layer)
    '''
    
    def __init__(self, vocab_size, hidden_size):
        '''
        Constructor for Encoder class.
        :param vocab_size: The size of the input language vocabulary
        :param hidden_size: The dimensionality to which input tokens will be embedded.
                            This in turn determines the dimensionality of the RNN module.
        '''
        
        super(Encoder, self).__init__()
        # Create the embedding layer
        self.embedding = Embedding(vocab_size, hidden_size)
        # Create the RNN layer (LSTM)
        # We also want the hidden/cell states of the LSTM to be returned, to feed into the decoder.
        self.rnn = LSTM(hidden_size, return_state=True)

        
    def call(self, x):
        '''
        Layer forward call() function.
        :param x: The input sequence of shape (batch_size, time_steps)
        
        :return: A tuple (e_h, e_h, e_c) Representing the (encoder output, encoder final hidden state, encoder final cell state)
                 The shape of each item is (batch_size, hidden_size)
        
        Note that in this module, we are not returning a sequence. Therefore the encoder output=encoder final hidden state (which
        makes the first item in this tuple redundant, but we return it anyway).
        '''
        
        x = self.embedding(x)
        x = self.rnn(x) # x is a tuple of (encoder output, encoder final hidden state, encoder final cell state)

        return x
    
    
class Decoder(tf.keras.layers.Layer):
    '''
    The decoder section of our NMT model. 
    
    It will consist of:
    1. An embedding layer to project the target language words (tokens) to hidden_size
    2. An RNN layer to generate the target language sequence (in this case, we use the LSTM layer)
    3. A fully connected layer to project the output tokens from hidden_size to (target language) vocab_size
    '''
    
    def __init__(self, vocab_size, hidden_size):
        '''
        TODO:
        Constructor for Decoder class.
        :param vocab_size: The size of the target language vocabulary
        :param hidden_size: The dimensionality to which tokens will be embedded.
                            This in turn determines the dimensionality of the RNN module.
        '''
        
        super(Decoder, self).__init__()
        ##########################################################################
        # TODO: Complete the __init__ constructor function similar to the Encoder class.     #
        # 
        # The Decoder must have the following attributes:
        # 1. An embedding layer to project the target language words (tokens) to hidden_size
        #
        # 2. An RNN layer to generate the target language sequence (in this case, we use the LSTM layer)
        #     NOTE: the LSTM layer in the Decoder is different to the encoder (we need a sequence output,
        #     and we DON'T want the final states of the LSTM as output like in the encoder). 
        #     Set the arguments of this layer carefully. (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
        #
        # 3. A fully connected layer to project the output tokens from hidden_size to (target language) vocab_size
        #
        ##########################################################################
        
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.rnn = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,  
            return_state=True       
        )
        self.fc = tf.keras.layers.Dense(units=vocab_size)
        
        ##########################################################################
        # END TODO
        ##########################################################################
        

    def call(self, x, encoder_state):
        '''
        TODO:
        Layer forward call() function.
        :param x: The target sequence of shape (batch_size, time_steps). During training, this will be the ground truth sequence.
                  During inference, this will just be the [SOS] start token (batched) since we do not have the ground truth.
        :param encoder_state: A tuple (e_h, e_c) representing the (encoder final hidden state, encoder final cell state) obtained
                  from the encoder module.
        
        :return: A sequence of model prediction outputs of shape (batch_size, time_steps, vocab_size)
        
        Note that in this module, we ARE returning a sequence. The dimensionality of each token in the sequence will be vocab_size
        (due to the fully connected layer).
        '''
        
        ##########################################################################
        # TODO: Complete the call() function similar to the Encoder class.     #
        # 
        # Perform the following steps in order:
        # 1. Feed input through the embedding layer. (remember we are embedding the target sequence now)
        #
        # 2. Feed through the RNN layer (LSTM in our case).
        #     NOTE: Don't forget to pass in the initial state of this layer based on the 
        #           encoder states
        #     https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        #
        # 3. Feed through the fully connected layer.
        #
        ##########################################################################
        x = self.embedding(x)  # Shape: (batch_size, time_steps, hidden_size)
        
        rnn_output, _, _ = self.rnn(x, initial_state=encoder_state)
        
        logits = self.fc(rnn_output)
        
        ##########################################################################
        # END TODO
        ##########################################################################
        
        
        return logits
    
class TranslationModel(tf.keras.Model):
    '''
    The Translation Model consists of the encoder and decoder modules.
    The model will behave differently depending on whether it is called during training or inference (whether to use teacher forcing or not).
    '''
    
    def __init__(self, 
                 nl_vocab_size,
                 eng_vocab_size,
                 hidden_size,
                 eng_vocab,
                 max_length = 30,
                 bidirectional_encoder = False):
        '''
        Constructor for TranslationModel class.
        :param nl_vocab_size: The size of the input language vocab (int)
        :param eng_vocab_size: The size of the target language vocab (int)
        :param hidden_size: The dimensionality of the encoder/decoder/embedding modules (int)
        :param eng_vocab: The target language vocabulary (dict)
        :param max_length: The max length of sequences (int) (Default 30)
        
        Note that the input language is already batched and padded to max_length.
        '''
        super(TranslationModel, self).__init__()
        
        # Initialize encoder/decoder attributes
        if not bidirectional_encoder:
            self.encoder = Encoder(nl_vocab_size, hidden_size)
        else:
            """
            **BONUS**
            Keep the parameter bidirectional_encoder to False until you do the bonus task (part 6).
            """
            self.encoder = BidirectionalEncoder(nl_vocab_size, hidden_size)
            
        self.decoder = Decoder(eng_vocab_size, hidden_size)
        
        self.max_length = max_length
        
        self.eng_vocab = eng_vocab
        self.token_to_eng = {val: key for key, val in eng_vocab.items()}
        self.eng_start_token_index = self.token_to_eng['[SOS]']
        self.eng_end_token_index = self.token_to_eng['[EOS]']
        self.eng_pad_token_index = self.token_to_eng['[PAD]']

        
    def call(self, nl_sequence, eng_sequence = None, training = True):
        '''
        TODO:
        Model forward call() function.
        :param nl_sequence: The input language sequence of shape (batch_size, time_steps). 
        :param eng_sequence: The target language sequence of shape (batch_size, time_steps). 
                  This will only be provided during training. (Default None)
        :param training: Boolean value indicating if the model is being used in training or not.
        
        :return: A sequence of model prediction outputs of shape (batch_size, time_steps, vocab_size)
        '''
        ##########################################################################
        # TODO: Complete the sections of this function.                          #
        # You are already provided the code for inference. You just need to      #
        # complete the required sections.                                        #
        ##########################################################################
        
        ####################################################
        # TODO: Feed in the input language sequence to the #
        # Encoder layer. Remember that the Encoder outputs #
        # a tuple (e_h, e_h, e_c)                          #
        ####################################################
        encoder_output, encoder_h, encoder_c = self.encoder(nl_sequence)

        
        ####################################################
        # END TODO                                         #
        ####################################################
        
        
        if training:
            ####################################################
            # TODO: Feed in the target language sequence and   #
            # encoder states to the decoder. Save it to the    #
            # 'output' variable.                                #
            ####################################################
            
            output = self.decoder(eng_sequence, (encoder_h, encoder_c))
            
            ####################################################
            # END TODO                                         #
            ####################################################
            
        else:
            # When not in training mode, we will need to generate a tensor of [SOS] token values to feed into the decoder.
            batch_size = nl_sequence.shape[0]
            input_sequence = tf.cast(tf.fill((batch_size, 1), value = self.eng_start_token_index), tf.int64)#tf.ones((batch_size,1),  dtype=tf.dtypes.int64) 
            
            output = tf.zeros((batch_size,0),  dtype=tf.dtypes.int64)
            
            # Perform the prediction one timestep at a time until we reach the max_length
            for time_step in range(self.max_length):
                
                decoder_out = self.decoder(input_sequence, (encoder_h, encoder_c)) 
                
                prediction = tf.argmax(decoder_out[:,-1,:], -1)
                
                output = tf.concat([output, tf.expand_dims(prediction, axis=1)], axis = 1)
                input_sequence = tf.concat([input_sequence, tf.expand_dims(prediction, axis=1)], axis = 1)
            
        return output
                    

        
    def decode_tokens(self, output_tokens):
        '''
        Utility function to decode the output sequence.
        :param output_tokens: The output sequence to be decoded of shape (batch_size, time_steps).
        
        :return: the list of decoded sentences
        '''
        all_decoded_sentences = []
        
        for sentence in output_tokens:
            decoded_sentence = []
            for token in sentence:
                if token == self.eng_end_token_index:
                    break
                decoded_sentence.append(self.eng_vocab[token.numpy()])

            all_decoded_sentences.append(decoded_sentence)
        
        return all_decoded_sentences
    


class BidirectionalEncoder(tf.keras.layers.Layer):
    '''
    **BONUS**
    The bidirectional encoder section of our NMT model. 
    
    It will consist of:
    1. An embedding layer to project the input language words (tokens) to hidden_size
    2. An RNN layer to encode the sequence (in this case, we use the LSTM layer)
    '''
    
    def __init__(self, vocab_size, hidden_size):
        '''
        Constructor for Encoder class.
        :param vocab_size: The size of the input language vocabulary
        :param hidden_size: The dimensionality to which input tokens will be embedded.
                            This in turn determines the dimensionality of the RNN module.
        '''
        
        super(BidirectionalEncoder, self).__init__()
        # Create the embedding layer
        self.embedding = Embedding(vocab_size, hidden_size)

        ##########################################################################
        # **BONUS**
        # TODO: Complete the __init__ constructor function similar to the Encoder class by defining a new
        # bidirectional rnn for self.rnn
        # 
        ##########################################################################
        # Create the bidirectional RNN layer (LSTM)
        # We also want the hidden/cell states of the LSTM to be returned, to feed into the decoder.
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True),
            merge_mode=None)                

        
        ##########################################################################
        # END TODO
        ##########################################################################
        

        
    def call(self, x):
        '''
        Layer forward call() function.
        :param x: The input sequence of shape (batch_size, time_steps)
        
        :return: A tuple (e_h, e_h, e_c) Representing the (encoder output, encoder final hidden state, encoder final cell state)
                 The shape of each item is (batch_size, hidden_size)
        
        Note that in this module, we are not returning a sequence. Therefore the encoder output=encoder final hidden state (which
        makes the first item in this tuple redundant, but we return it anyway).
        '''
        
        x = self.embedding(x)
        outputs = self.rnn(x)

        encoder_lstm_x = outputs[0]  
        enc_state_h_fwd = outputs[1]  
        enc_state_c_fwd = outputs[2]  
        enc_state_h_bwd = outputs[3]  
        enc_state_c_bwd = outputs[4]  

        x = (encoder_lstm_x, enc_state_h_fwd + enc_state_h_bwd, enc_state_c_fwd + enc_state_c_bwd) 

        return x