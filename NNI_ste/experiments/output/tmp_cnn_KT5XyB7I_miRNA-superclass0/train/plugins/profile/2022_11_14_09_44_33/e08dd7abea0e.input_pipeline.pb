	,���c�@,���c�@!,���c�@	��@0@��@0@!��@0@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL,���c�@MJA��4�?1��m���?AWya��?I ���	@Y�F���R�?rEagerKernelExecute 0*	S㥛�Ju@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���+,�?!��܆�K@)5�ׂ��?1�l:LJ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���@��?!��Q��8@)gI-�L�?1�4kw�4@:Preprocessing2F
Iterator::ModelIM��f��?!喘5+@)A�S����?1*�ׅ@�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��PMI��?!#�L�]�U@)�ht�3�?1�G5�O@:Preprocessing2U
Iterator::Model::ParallelMapV2P�"�Ɣ?!�cY���@)P�"�Ɣ?1�cY���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Q�d=�?!��P�Z@)�Q�d=�?1��P�Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensori��Q��?!��R��@)i��Q��?1��R��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�V|Cᳵ?!�\�3��8@)����g?16��3w�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�54.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t25.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��@0@IDd�a�T@Q���-@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	MJA��4�?MJA��4�?!MJA��4�?      ��!       "	��m���?��m���?!��m���?*      ��!       2	Wya��?Wya��?!Wya��?:	 ���	@ ���	@! ���	@B      ��!       J	�F���R�?�F���R�?!�F���R�?R      ��!       Z	�F���R�?�F���R�?!�F���R�?b      ��!       JGPUY��@0@b qDd�a�T@y���-@