	��X���K@��X���K@!��X���K@	�񿏮@�񿏮@!�񿏮@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0��X���K@OqN`�?1x%�s}�?@Ix�g�U3@Y�X5s�@r0*	��"��i@2F
Iterator::Model*��Dش?!
gajBD@)�
(�ӧ?1� 4�+(7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���<�?!1-�bW9@)
h"lxz�?1 m8��4@:Preprocessing2U
Iterator::Model::ParallelMapV2Pqx�ܡ?!M͎�\1@)Pqx�ܡ?1M͎�\1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q75М?!���� ,@)�q75М?1���� ,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{Cr�?!k�9@)N���P�?1�V�$�f'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�9w�^��?!�����M@)�X�|^�?1��[L[� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(b�c�?!� SO��@)(b�c�?1� SO��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�35.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�񿏮@I����x�A@Q�KB���L@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	OqN`�?OqN`�?!OqN`�?      ��!       "	x%�s}�?@x%�s}�?@!x%�s}�?@*      ��!       2      ��!       :	x�g�U3@x�g�U3@!x�g�U3@B      ��!       J	�X5s�@�X5s�@!�X5s�@R      ��!       Z	�X5s�@�X5s�@!�X5s�@b      ��!       JGPUY�񿏮@b q����x�A@y�KB���L@