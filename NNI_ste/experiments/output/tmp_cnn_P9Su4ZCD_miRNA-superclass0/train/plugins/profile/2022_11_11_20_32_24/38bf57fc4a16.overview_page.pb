�	{ܷZ'� @{ܷZ'� @!{ܷZ'� @	qD���@qD���@!qD���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL{ܷZ'� @x����l�?1=�|���@A���}��?I� :�@Y5�8EGr�?rEagerKernelExecute 0*	?5^�I�d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$D���?!���E@)? �M�ܯ?1Z����B@:Preprocessing2F
Iterator::ModelZ_&���?!/1�Z�?@)�|�H�F�?1a�m�]3@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice� Z+��?!ˉ�̀+@)� Z+��?1ˉ�̀+@:Preprocessing2U
Iterator::Model::ParallelMapV2_%���?!̛�J�|(@)_%���?1̛�J�|(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#��~j��?!@�3[�Q@)��HV�?1����%�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�}��A�?!ZeR&ш@)�}��A�?1ZeR&ш@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�v���?!~�U�p2@)��o��?1�{��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�.�H��?!y�Y"/4@)�%��og?1�ϧ4���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 22.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�31.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9pD���@I�d�3K@Q��bF��D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	x����l�?x����l�?!x����l�?      ��!       "	=�|���@=�|���@!=�|���@*      ��!       2	���}��?���}��?!���}��?:	� :�@� :�@!� :�@B      ��!       J	5�8EGr�?5�8EGr�?!5�8EGr�?R      ��!       Z	5�8EGr�?5�8EGr�?!5�8EGr�?b      ��!       JGPUYpD���@b q�d�3K@y��bF��D@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�Z͒f�?!�Z͒f�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�s�H�?!��R����?0"1
model/Conv1D_3/conv1dConv2DH��BD�?!����3M�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput���<��?!O��p�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputl�X]jЭ?!䀼��?0"1
model/Conv1D_2/conv1dConv2D�3��ҥ?!�JCg��?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradZ�1K;�?!�d��&�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad#��?!�F��Õ�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad-+�k�?!��w�{��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad��D��i�?!iJ4C�?Q      Y@Y�a�2�t'@a�ӭ�aV@q/9:D�=@y
N���S�?"�
both�Your program is POTENTIALLY input-bound because 22.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�31.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�29.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 