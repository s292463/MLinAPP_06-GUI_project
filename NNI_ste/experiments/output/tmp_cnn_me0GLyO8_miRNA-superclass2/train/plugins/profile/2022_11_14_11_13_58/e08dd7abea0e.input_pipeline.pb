	�h㈵�(@�h㈵�(@!�h㈵�(@	I8�/��@I8�/��@!I8�/��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�h㈵�(@)��Pj�@1���� @AٙB�5v�?IA�C��?Yxb֋���?rEagerKernelExecute 0*	��"���h@2F
Iterator::Modelp�x�0�?!�ia���C@)g*��?10b�ǕB<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� >��?!��yk��7@)�MbX9�?1|�c/R4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#J{�/L�?!
��?N@)?���2�?1��$s��-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice֩�=#�?!*�J���)@)֩�=#�?1*�J���)@:Preprocessing2U
Iterator::Model::ParallelMapV2C�up��?!w�f�u'@)C�up��?1w�f�u'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev�1<��?!��YN��3@)Ǆ�K���?1v@�r@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorb��m�R}?!w2��9
@)b��m�R}?1w2��9
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�V_]��?!x�0`�r5@)����k?1M�r���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�13.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9I8�/��@I��})�8@@QK�QLP@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	)��Pj�@)��Pj�@!)��Pj�@      ��!       "	���� @���� @!���� @*      ��!       2	ٙB�5v�?ٙB�5v�?!ٙB�5v�?:	A�C��?A�C��?!A�C��?B      ��!       J	xb֋���?xb֋���?!xb֋���?R      ��!       Z	xb֋���?xb֋���?!xb֋���?b      ��!       JGPUYI8�/��@b q��})�8@@yK�QLP@