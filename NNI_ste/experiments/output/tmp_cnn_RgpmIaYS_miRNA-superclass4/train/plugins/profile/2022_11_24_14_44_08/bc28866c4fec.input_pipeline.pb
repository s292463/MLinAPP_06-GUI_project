	�����3@�����3@!�����3@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�����3@� �6q!@1�O!#@Ap>u�Rz�?IRb��v�?rEagerKernelExecute 0*	Zd;�3u@2U
Iterator::Model::ParallelMapV2Xs�`��?!�)W��L@)Xs�`��?1�)W��L@:Preprocessing2F
Iterator::Model���i�?!&���4�R@)�ʃ�9�?1ِu�?0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����k�?!HE�]��*@)uWv����?1Iz=��&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/0+�~�?!j��-�9@)3��bb�?1b4��k�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$�����?!u��y�@)$�����?1u��y�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateE���V	�?!F�T��_@)e���V�?1��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorOt	�~?!�+�^�@)Ot	�~?1�+�^�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(��0�?!����~@)A��h:;i?1���ӑ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 42.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��z��I@QE(��cH@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� �6q!@� �6q!@!� �6q!@      ��!       "	�O!#@�O!#@!�O!#@*      ��!       2	p>u�Rz�?p>u�Rz�?!p>u�Rz�?:	Rb��v�?Rb��v�?!Rb��v�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��z��I@yE(��cH@