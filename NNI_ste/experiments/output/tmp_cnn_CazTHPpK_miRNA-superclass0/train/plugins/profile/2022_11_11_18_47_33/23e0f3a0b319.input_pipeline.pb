	�{��C@�{��C@!�{��C@	�׸C��@�׸C��@!�׸C��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�{��C@�9z��&�?1`��A�?Ab��U��?I�,��@YRI��&��?rEagerKernelExecute 0*	L7�A``@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�6����?!H�[{�@@)�7k�?1qrn �<@:Preprocessing2F
Iterator::Model�!�A�?!ꐇl�	F@)�ʉv�?1u�m���;@:Preprocessing2U
Iterator::Model::ParallelMapV2Z�����?!_��=s�0@)Z�����?1_��=s�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���4�?!�ݜKt�@)���4�?1�ݜKt�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipg��I}Y�?!ox�g�K@)X�ۼq�?1��)[@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��T���?!��y-�*@).2�~?1MXB���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor܂����y?!�Օ#�@)܂����y?1�Օ#�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9{g�UI�?!Մ�J�.@)�:���Re?1I��BW? @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�55.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�׸C��@I���A0S@Q'�<�h4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�9z��&�?�9z��&�?!�9z��&�?      ��!       "	`��A�?`��A�?!`��A�?*      ��!       2	b��U��?b��U��?!b��U��?:	�,��@�,��@!�,��@B      ��!       J	RI��&��?RI��&��?!RI��&��?R      ��!       Z	RI��&��?RI��&��?!RI��&��?b      ��!       JGPUY�׸C��@b q���A0S@y'�<�h4@