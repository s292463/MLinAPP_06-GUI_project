	�b��`%@�b��`%@!�b��`%@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�b��`%@��u�|�?1�&�ʰ@A�'��9x�?I!XU/��@rEagerKernelExecute 0*	L7�A`yc@2F
Iterator::Model"��pӱ?!Ki�C~XF@)�Md���?1)&9U�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Z}uU��?!]����]<@)`����?1[�}��6@:Preprocessing2U
Iterator::Model::ParallelMapV2{���?!�X�c(<.@){���?1�X�c(<.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��^
�?!��m���K@)<�H��ڒ?1���EE�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���/g�?!
�+@)���/g�?1
�+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���8��?!��KN��)@)�"[Aӂ?1[9�p��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ȭI�%�?!\1
�-�@)�ȭI�%�?1\1
�-�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap衶� �?!d�$�q?.@)X�\Tl?1�Ze�E�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�63.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIxO�]*�S@Q�V�5@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��u�|�?��u�|�?!��u�|�?      ��!       "	�&�ʰ@�&�ʰ@!�&�ʰ@*      ��!       2	�'��9x�?�'��9x�?!�'��9x�?:	!XU/��@!XU/��@!!XU/��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qxO�]*�S@y�V�5@