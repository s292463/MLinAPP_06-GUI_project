	�:����8@�:����8@!�:����8@	�����w,@�����w,@!�����w,@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�:����8@<�ݭ,�y?1 Sh�@I%=�N�2@Y���QI]@r0*	���Mbj@2F
Iterator::ModelX�%����?!-���*F@)��e�c]�?1�t�z��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�@1�d�?!~J��lo<@)�2���?1�p�78@:Preprocessing2U
Iterator::Model::ParallelMapV2�����P�?!R#+�+m+@)�����P�?1R#+�+m+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�8�~߿�?!�|�X8�K@)c&Q/�4�?1�Rv�8�%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!�����?!Å���_0@)�Pk�w��?1�v�&�i!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�C�ͩd�?!�)��@)�C�ͩd�?1�)��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor섗���?!�5�t�@)섗���?1�5�t�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 14.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�75.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�����w,@I�u�[��R@Q���b��#@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	<�ݭ,�y?<�ݭ,�y?!<�ݭ,�y?      ��!       "	 Sh�@ Sh�@! Sh�@*      ��!       2      ��!       :	%=�N�2@%=�N�2@!%=�N�2@B      ��!       J	���QI]@���QI]@!���QI]@R      ��!       Z	���QI]@���QI]@!���QI]@b      ��!       JGPUY�����w,@b q�u�[��R@y���b��#@