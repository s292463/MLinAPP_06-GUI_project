	:�8@:�8@!:�8@	B���Ǳ@B���Ǳ@!B���Ǳ@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL:�8@yv�֧@1��
~[ @Aۤ���w�?I�R?o*�@Yew�h��?rEagerKernelExecute 0*	B`��"c@2F
Iterator::ModelY|E��?! �!(I�C@)��0|D�?1g���C�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Y����?!n_
�q�;@)�}"O�?1%���t7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�z�f�l�?!%3�b�):@)0��DK�?1�Ўg3@:Preprocessing2U
Iterator::Model::ParallelMapV2np���?!4��Ҝ�*@)np���?14��Ҝ�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��W\�?!"��P�
@)��W\�?1"��P�
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-Z��լ�?!� �׶SN@)ڨN�~?16����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|~!<z?!'e:ď�@)|~!<z?1'e:ď�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapl?���?!����<@)G�ŧ h?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 29.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�39.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9A���Ǳ@I���XQ@Q���[�);@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	yv�֧@yv�֧@!yv�֧@      ��!       "	��
~[ @��
~[ @!��
~[ @*      ��!       2	ۤ���w�?ۤ���w�?!ۤ���w�?:	�R?o*�@�R?o*�@!�R?o*�@B      ��!       J	ew�h��?ew�h��?!ew�h��?R      ��!       Z	ew�h��?ew�h��?!ew�h��?b      ��!       JGPUYA���Ǳ@b q���XQ@y���[�);@