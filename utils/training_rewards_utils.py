import tensorflow as tf


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)

    loss = tf.reduce_mean(neg_logprob * rewards)  # TODO

    return loss


def train_step(model, loss_function, optimizer, observations, actions, discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn(observations)
        else:
            prediction = model(observations)
        loss = loss_function(prediction, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)  # TODO
    grads, _ = tf.clip_by_global_norm(grads, 2)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def compute_driving_loss(dist, actions, rewards):
    neg_logprob = -1 * dist.log_prob(actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss
