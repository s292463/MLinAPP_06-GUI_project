	I/j���'@I/j���'@!I/j���'@	U�ڦ�@U�ڦ�@!U�ڦ�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLI/j���'@�Q��?179|҉�"@Ab��BW"�?Iis�ۄ��?Y}�|�?rEagerKernelExecute 0*	�E����u@2U
Iterator::Model::ParallelMapV2)��/���?!�V]��$I@))��/���?1�V]��$I@:Preprocessing2F
Iterator::Modelm���"�?!�c 1��P@)�<��@�?1��WS�0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��� ��?!ayBc�%@)��΢w*�?1����"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�L��Ӏ�?!�8��}n@@)���/�Ɯ?1�i�Ǹ @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateu ��W�?!`�t�d *@)�/fKVE�?1�ot�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice
pUj�?!�t٠��@)
pUj�?1�t٠��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�h9�Cm{?!��*���?)�h9�Cm{?1��*���?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\kF�?! ҙ��+@)k�3�j?1=���`��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9U�ڦ�@I�c*pj[/@Q=>MD��S@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Q��?�Q��?!�Q��?      ��!       "	79|҉�"@79|҉�"@!79|҉�"@*      ��!       2	b��BW"�?b��BW"�?!b��BW"�?:	is�ۄ��?is�ۄ��?!is�ۄ��?B      ��!       J	}�|�?}�|�?!}�|�?R      ��!       Z	}�|�?}�|�?!}�|�?b      ��!       JGPUYU�ڦ�@b q�c*pj[/@y=>MD��S@