'''
E4040 2024Fall Assignment3
Machine Translation
'''
import tensorflow as tf

def masked_loss(y_true, y_pred):
    """
    Loss function that computes Cross Entropy while ignoring pad tokens.
    (We don't want to penalize our model based on pad tokens)
    :param y_true: The ground truth sequence
    :param y_pred: The predicted sequence
    
    :return: The computed loss value
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def train_seq2seq_model(model, train_loader, optimizer, num_epochs):
    '''
    Utility function to train the seq2seq model.
    
    :param model: the tf.keras.Model to train.
    :param train_loader: the tf.data.Dataset to load train data. It must already be batched.
    :param optimizer: The tf.keras.optimizers.Optimizer algorithm to use. 
    :param num_epochs: The number of epochs to train (int).
    '''

    losses = []
    # Iterate epochs
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        # Iterate batches
        for step, (x_train, y_train) in enumerate(train_loader):
            # Remove [EOS] token from input, as we want to end prediction when we see this, and we don't want to predict for this input
            eos_mask = y_train != 3 #The EOS token value=3
            num_elements = y_train.shape[0] * y_train.shape[1]
            samples_without_eos = tf.math.count_nonzero(eos_mask)

            # If there are sentences without EOS in the batch, add the EOS for these sentences and PAD for the rest.
            if samples_without_eos == (num_elements - y_train.shape[0]):
                y_inputs_no_eos = tf.reshape(y_train[eos_mask], (y_train.shape[0], y_train.shape[1]-1))
            else:
                # Add extra padding and EOS
                extra_column = []
                for row in y_train:
                    if False in row:
                        extra_column.append(0) #The pad token value=0
                    else:
                        extra_column.append(3) #The EOS token value=3

                y_train = tf.concat([y_train, tf.expand_dims(tf.constant(extra_column, dtype = tf.int64), axis=1)], axis = 1)
                eos_mask = y_train != 3
                y_inputs_no_eos = tf.reshape(y_train[eos_mask], (y_train.shape[0], y_train.shape[1]-1))

            # Remove [SOS] token from target, as we do not want to predict it.
            shift_y_train_target = y_train[:, 1:]
            
            # Perform forward pass with GradientTape
            with tf.GradientTape() as tape:
                decoder_out = model(x_train, y_inputs_no_eos) #Model feedforward
                loss_value = masked_loss(shift_y_train_target, decoder_out) #Compute loss
            
            # Calculate gradients
            grads = tape.gradient(loss_value, model.trainable_weights)
            # Apply gradients to weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            losses.append(loss_value)

            if step % 50 == 0:
                print(f"Iter: {step}, Loss (iter): {loss_value}, Mean Loss (over last 50 iters): {tf.math.reduce_mean(losses[-50:])}")

