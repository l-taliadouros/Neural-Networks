'''
E4040 2024Fall Assignment3
Machine Translation
'''
import tensorflow as tf

def get_dataset(nl_text, eng_text):
    '''
    Utility function to build a tf Dataset from the text data. 

        :param nl_text: a numpy.ndarray of shape [num_sentences, max_sentence_len],
                        containing the dutch data
        :param eng_text: a numpy.ndarray of shape [num_sentences, max_sentence_len],
                        containing the eng data

        :return: a tf.data.Dataset object built from the text data.
    '''
    ###################################################
    # TODO: Create the dataset object. Be careful of  #
    # the input language and target language.         #
    ###################################################

    text_ds = tf.data.Dataset.from_tensor_slices((nl_text, eng_text))

    
    ###################################################
    # END TODO                                        #
    ###################################################
    
    return text_ds



def get_dataset_partitions_tf(
    ds, 
    ds_size, 
    val_split=0.1, 
    shuffle=True, 
    shuffle_size=10000):
    """
    Split a dataset into training, validation, and test sets.

    :param ds: The input dataset. (tf.data.Dataset)
    :param ds_size: The total number of elements in the dataset.
    :param val_split: The fraction of the dataset to allocate to the validation set. Default is 0.1.
    :param shuffle: If True, shuffle the dataset before splitting. Default is True.
    :param shuffle_size: The buffer size for shuffling the dataset. Default is 10000.

    :return: Two tf.data.Dataset objects representing the training, and validation sets.
    """
    assert val_split <= 1.
    
    #################################################################
    # TODO: Split ds into train_ds and val_ds.
    # 1. Shuffle input ds based on the shuffle argument.
    #
    # 2. Split ds into train_ds and val_ds
    # - Use the split fraction arguments to determine
    # the size of the resulting datasets.
    # - You may want to use dataset.take() and
    # dataset.skip() functions to create your datasets
    # Refer to:
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip
    #
    # NOTE: Take care of the variable names of the new datasets,
    # make sure they are exactly like above.
    # 
    #################################################################
    train_size = int(ds_size * (1 - val_split))
    val_size = ds_size - train_size
    
    if shuffle:
        #Shuffle dataset
        #Remember to use the shuffle_size argument for buffering.
        #You may want to set a seed for reproducibility, else the shuffle
        #will result in different splits every time it is called.
        ds = ds.shuffle(shuffle_size, seed=42)  # Shuffle with a fixed seed for reproducibility

    # Create train and validation datasets
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)
    
    #################################################################
    # END TODO                                                      #
    #################################################################
    
    
    return train_ds, val_ds


def decode_text(text, vocab):
    """
    Decode a sequence of integers into a list of tokens using a vocabulary mapping.

    :param text: A list of integer tokens to be decoded.
    :param vocab: A dictionary that maps integer tokens to their corresponding tokens or words.

    :return: A list of tokens representing the decoded text.
    """
    decoded_text = [vocab[i] for i in text]
    return decoded_text

