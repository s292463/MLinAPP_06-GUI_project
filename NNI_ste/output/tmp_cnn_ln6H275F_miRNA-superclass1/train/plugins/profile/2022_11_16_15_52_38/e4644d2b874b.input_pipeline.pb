	Gsd�W@Gsd�W@!Gsd�W@	Lk�d�@Lk�d�@!Lk�d�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLGsd�W@����9�?1b�oO @A��R{m�?Ix{�@Y�d�?rEagerKernelExecute 0*	������f@2F
Iterator::Model���\�ظ?!���v��J@)C;�Y�ݩ?1     <@:Preprocessing2U
Iterator::Model::ParallelMapV2���_Zԧ?!����9@)���_Zԧ?1����9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���f��?!���1<@)^�SH�?1��J�8@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice������?!��;[4@)������?1��;[4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate8��+ؖ?!CG�;��(@)y�[Y��?1�p�;A@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ��WW�?!zv4�$G@))A�G�~?1��P~ǈ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%u�~?!�s+�{K@)%u�~?1�s+�{K@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǂ L��?!��}5��+@)mU�Yf?1'R4�1�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�45.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Mk�d�@IX����ZQ@Q9a��P�:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����9�?����9�?!����9�?      ��!       "	b�oO @b�oO @!b�oO @*      ��!       2	��R{m�?��R{m�?!��R{m�?:	x{�@x{�@!x{�@B      ��!       J	�d�?�d�?!�d�?R      ��!       Z	�d�?�d�?!�d�?b      ��!       JGPUYMk�d�@b qX����ZQ@y9a��P�:@