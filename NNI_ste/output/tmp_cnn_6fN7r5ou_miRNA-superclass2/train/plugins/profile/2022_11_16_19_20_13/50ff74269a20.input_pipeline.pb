	;6��!@;6��!@!;6��!@	��7/�$!@��7/�$!@!��7/�$!@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL;6��!@p$�`Sg�?1�9]�@A�eo)�?I�"��?Y7���a�?rEagerKernelExecute 0*	�ZdGf@2F
Iterator::Model�4~�$�?!X�j�@�D@)�$]3�f�?1a�@H>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!�E��P9@)�t"�T�?1:#^�	/5@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�b��	��?!R��fX2@)�b��	��?1R��fX2@:Preprocessing2U
Iterator::Model::ParallelMapV2��>eĕ?!����r�'@)��>eĕ?1����r�'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�k*�¦?!BL�P��8@)ͫ:���?1Ļv�oZ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip5��{�?!�
�-�M@)aE|�?1*�mr)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU���)~?!Љl���@)U���)~?1Љl���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��8�Z�?!Z�_�:@)1�䠄i?1f�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�20.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t15.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��7/�$!@I��p��A@QC>�+�K@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	p$�`Sg�?p$�`Sg�?!p$�`Sg�?      ��!       "	�9]�@�9]�@!�9]�@*      ��!       2	�eo)�?�eo)�?!�eo)�?:	�"��?�"��?!�"��?B      ��!       J	7���a�?7���a�?!7���a�?R      ��!       Z	7���a�?7���a�?!7���a�?b      ��!       JGPUY��7/�$!@b q��p��A@yC>�+�K@