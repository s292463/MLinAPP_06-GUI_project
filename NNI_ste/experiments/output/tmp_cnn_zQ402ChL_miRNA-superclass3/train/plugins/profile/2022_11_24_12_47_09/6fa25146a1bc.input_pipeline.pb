	��tw�U @��tw�U @!��tw�U @      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��tw�U @#M�<i�?1�u�r���?A�-��T�?I8�Q��}@rEagerKernelExecute 0*	أp=
�c@2F
Iterator::Model��K�A��?!��g��BH@)�_u�Hg�?1�?ȣ>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat.Ȗ��2�?!>�5g�8@)����˚�?1L���r4@:Preprocessing2U
Iterator::Model::ParallelMapV2M�D�u��?!��x2@)M�D�u��?1��x2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�B�l�?!C1�$�I@)W��Ma��?19A9&��&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���D��?!���T�E@)���D��?1���T�E@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�x$^�Ε?!��Ά�*@)i�-���?1���9p@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�̯� �|?!�����@)�̯� �|?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���y��?![ջQ�=.@)��f��e?1UhW~�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�59.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI±`՚�S@Q�8}��I4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#M�<i�?#M�<i�?!#M�<i�?      ��!       "	�u�r���?�u�r���?!�u�r���?*      ��!       2	�-��T�?�-��T�?!�-��T�?:	8�Q��}@8�Q��}@!8�Q��}@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q±`՚�S@y�8}��I4@