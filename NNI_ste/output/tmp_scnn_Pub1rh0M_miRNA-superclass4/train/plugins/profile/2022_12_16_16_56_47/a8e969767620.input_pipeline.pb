	���9#26@���9#26@!���9#26@	����> @����> @!����> @"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���9#26@�U��;�?1 {�\.$@II����$@Y�Xİ���?r0*	]d;�O�i@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�}�<�?!AI7v�E@)tF��_�?1]n��R?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�:�� �?!H���=@)���<HO�?13��)P68@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��(��P�?!�'�l�,)@)��(��P�?1�'�l�,)@:Preprocessing2U
Iterator::Model::ParallelMapV2�����?!L&���(@)�����?1L&���(@:Preprocessing2F
Iterator::Model2�]�)ʥ?!�[�D�4@)�tp�x�?1���羶 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��b)���?! i���S@)}%����?1��.mz�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensori�wak�?!I[8r@)i�wak�?1I[8r@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�46.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����> @I��9%85G@Q��!v�F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�U��;�?�U��;�?!�U��;�?      ��!       "	 {�\.$@ {�\.$@! {�\.$@*      ��!       2      ��!       :	I����$@I����$@!I����$@B      ��!       J	�Xİ���?�Xİ���?!�Xİ���?R      ��!       Z	�Xİ���?�Xİ���?!�Xİ���?b      ��!       JGPUY����> @b q��9%85G@y��!v�F@