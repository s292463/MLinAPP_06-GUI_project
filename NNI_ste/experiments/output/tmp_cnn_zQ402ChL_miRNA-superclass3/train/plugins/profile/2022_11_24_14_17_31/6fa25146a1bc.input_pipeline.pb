	cC7��@cC7��@!cC7��@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCcC7��@U�3�Y�?1N�t"��?Aު�PMI�?I����@rEagerKernelExecute 0*	T㥛Ğs@2F
Iterator::Modeli6��`��?!߼Y�HS@)�$y���?1Q��aQ@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat${��!U�?!��з�L)@)Q��dV�?1e�a�%@:Preprocessing2U
Iterator::Model::ParallelMapV2�:TS�u�?!�(N4|o@)�:TS�u�?1�(N4|o@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��J&��?!5��0@)��J&��?15��0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipq�Ws�`�?!��v��6@)l�`q8�?1�ғc��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)����B�?!��aO6@)��B�iށ?1�9�<@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Y.{?!g�XA� @)����Y.{?1g�XA� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt��;�?!�J�8��@)\��b��g?1U۵L��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�52.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���g\S@Q�a��7@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	U�3�Y�?U�3�Y�?!U�3�Y�?      ��!       "	N�t"��?N�t"��?!N�t"��?*      ��!       2	ު�PMI�?ު�PMI�?!ު�PMI�?:	����@����@!����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���g\S@y�a��7@