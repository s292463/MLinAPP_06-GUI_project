	��L�Dt-@��L�Dt-@!��L�Dt-@	�,	X�9�?�,	X�9�?!�,	X�9�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��L�Dt-@d������?1}����'@AD��<��O?I\U�]|�?Y�lw��?rEagerKernelExecute 0*	�A`��\u@2U
Iterator::Model::ParallelMapV2��!����?!Y��r��J@)��!����?1Y��r��J@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~�f+�?!���Q��1@)����U��?1M�I�/@:Preprocessing2F
Iterator::Model�v��-u�?!AI�
zgQ@)7�����?1��k��y/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Kq�?!��q�g�@)����Kq�?1��q�g�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����o�?!�����!@)�x��?1i����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����镺?!��v�b>@)����qn�?1��M�5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory��"��{?!�:�\w�?)y��"��{?1�:�\w�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6����$�?!]U��#@)�Nw�x�f?1uA<u�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�,	X�9�?I��a
�0@Q�j�1�^T@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	d������?d������?!d������?      ��!       "	}����'@}����'@!}����'@*      ��!       2	D��<��O?D��<��O?!D��<��O?:	\U�]|�?\U�]|�?!\U�]|�?B      ��!       J	�lw��?�lw��?!�lw��?R      ��!       Z	�lw��?�lw��?!�lw��?b      ��!       JGPUY�,	X�9�?b q��a
�0@y�j�1�^T@