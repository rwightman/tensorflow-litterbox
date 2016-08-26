# UC Berkeley's Standard Copyright and Disclaimer Notice:
# 
# Copyright (c) 2016. The Regents of the University of California (Regents). All
# Rights Reserved. Permission to use, copy, modify, and distribute this software
# and its documentation for educational, research, and not-for-profit purposes,
# without fee and without a signed licensing agreement, is hereby granted,
# provided that the above copyright notice, this paragraph and the following
# two paragraphs appear in all copies, modifications, and distributions.
# Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue,
# Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, for commercial licensing
# opportunities.
# 
# Ronghang Hu, University of California, Berkeley.
# 
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
# THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
# THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
# PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling

    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.

    Returns:
        a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    rand_h = rand_h.astype(np.int64)
    rand_s = rand_s.astype(np.float32)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix

def compact_bilinear_pooling(bottom1, bottom2, output_dim, sum_pool=True,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7):
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:

    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).

    Args:
        bottom1: 1st input, 4D Tensor of shape [batch_size, height, width, input_dim1].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, height, width, input_dim2].

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.

    Returns:
        Compact bilinear pooled results of shape [batch_size, output_dim] or
        [batch_size, height, width, output_dim], depending on `sum_pool`.
    """

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    if rand_h_1 is None:
        np.random.seed(seed_h_1)
        rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    if rand_s_1 is None:
        np.random.seed(seed_s_1)
        rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)
    if rand_h_2 is None:
        np.random.seed(seed_h_2)
        rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    if rand_s_2 is None:
        np.random.seed(seed_s_2)
        rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])
    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
        bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
        bottom2_flat, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    zeros = tf.zeros_like(sketch1)
    fft1 = tf.batch_fft(tf.complex(real=sketch1, imag=zeros))
    fft2 = tf.batch_fft(tf.complex(real=sketch2, imag=zeros))

    # Step 3: Elementwise product
    fft_product = tf.mul(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(tf.batch_ifft(fft_product))
    output_shape = tf.add(tf.mul(tf.shape(bottom1), [1, 1, 1, 0]),
                          [0, 0, 0, output_dim])
    cbp = tf.reshape(cbp_flat, output_shape)

    # Step 5: Sum pool over spatial dimensions, if specified
    if sum_pool:
        cbp = tf.reduce_sum(cbp, reduction_indices=[1, 2])

    return cbp
