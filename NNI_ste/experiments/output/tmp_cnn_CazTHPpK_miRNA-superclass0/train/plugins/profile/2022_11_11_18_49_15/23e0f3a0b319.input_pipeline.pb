	�4}v�#@�4}v�#@!�4}v�#@	P{`N$��?P{`N$��?!P{`N$��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�4}v�#@�T��@1��B:|�?A}��A�<�?I���X�f@Y��~j�t�?rEagerKernelExecute 0*	��Q��p@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��(	���?!�����9@)��(	���?1�����9@:Preprocessing2F
Iterator::Modelfٓ���?!�@�2y=@)���*ø�?15�z�O4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�x`�?!�	'?P�1@)[��Ye��?1��X�B.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatei o�Ż?!s�*�XD@)vŌ�� �?1L��v-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3�ۃ�?!�/Js��Q@)���D-�?1G�̣`%@:Preprocessing2U
Iterator::Model::ParallelMapV2�.6��?!�wq�R"@)�.6��?1�wq�R"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�M)���}?!��L�@)�M)���}?1��L�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph"lxz��?!|1��%�D@)�?�߾l?1$!�-���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�59.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9P{`N$��?Io���5U@Q"U�5�<+@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�T��@�T��@!�T��@      ��!       "	��B:|�?��B:|�?!��B:|�?*      ��!       2	}��A�<�?}��A�<�?!}��A�<�?:	���X�f@���X�f@!���X�f@B      ��!       J	��~j�t�?��~j�t�?!��~j�t�?R      ��!       Z	��~j�t�?��~j�t�?!��~j�t�?b      ��!       JGPUYP{`N$��?b qo���5U@y"U�5�<+@