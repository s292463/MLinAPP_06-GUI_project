�	MHk:"@MHk:"@!MHk:"@	qFB��@qFB��@!qFB��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCMHk:"@(�.����?1�̔��@Ii�-���?Y�ND��?rEagerKernelExecute 0*	}?5^��e@2F
Iterator::Model�(ϼv�?!�j{���E@)�7�GnM�?1;�t�0n=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�A�Ѫ�?!L���\9@)�z1���?1I�6�D-5@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-�B;�Y�?!b�lK2@)-�B;�Y�?1b�lK2@:Preprocessing2U
Iterator::Model::ParallelMapV2�3�c�=�?!p��>,@)�3�c�=�?1p��>,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����B�?!$d�T��8@)B]¡�?17q@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��lu9�?!G��Kf9L@)8�Jw�ـ?14ˏe��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor7�����}?!
p�?��@)7�����}?1
p�?��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA�+���?!uq�-_:@)�h㈵�d?1�@-,w�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9qFB��@I�V"h^�3@Q"@��KlS@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	(�.����?(�.����?!(�.����?      ��!       "	�̔��@�̔��@!�̔��@*      ��!       2      ��!       :	i�-���?i�-���?!i�-���?B      ��!       J	�ND��?�ND��?!�ND��?R      ��!       Z	�ND��?�ND��?!�ND��?b      ��!       JGPUYqFB��@b q�V"h^�3@y"@��KlS@�"1
model/Conv1D_3/conv1dConv2D�]��?!�]��?"1
model/Conv1D_4/conv1dConv2Dro$�<�?!QG�OBz�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�ҽ:���?!J��
��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��3<�?!����w�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput!�M���?!ډ���)�?0"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam5f����?!P� ����?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�%��雘?!�����\�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��1��9�?!��pI��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput %}���?!Jx��A�?0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad%�wΕ?!l�(6���?Q      Y@Y���cj`'@a�O���V@q����X8@ys����?"�
both�Your program is POTENTIALLY input-bound because 6.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�12.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�24.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 