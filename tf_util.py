"""Some tensorflow utility functions"""

import tensorflow as tf


def ragged_boolean_mask(data, mask):
    """Masks each row individually, preserving row splits.

    There is a similar function ragged.boolean_mask in TF 2.0, although I'm not
    sure if it behaves exactly the same."""
    negated_mask_float = 1 - tf.cast(mask, tf.int64)
    # reduce_sum currently doesn't work for ragged tensors (as of 1.3).
    # Work around it by just filling the rows with 0s, it doesn't change
    # the sum.
    values_removed_per_row = tf.reduce_sum(
        negated_mask_float.to_tensor(), axis=1
    )
    old_row_lengths = data.row_lengths()
    masked_row_lengths = old_row_lengths - values_removed_per_row

    flat_values = data.values
    flat_mask = mask.values
    masked_values = tf.boolean_mask(flat_values, flat_mask)
    return tf.RaggedTensor.from_row_lengths(masked_values, masked_row_lengths)
