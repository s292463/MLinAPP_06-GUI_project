�	�#EdXu!@�#EdXu!@!�#EdXu!@	���F�@���F�@!���F�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�#EdXu!@UގpZ�@16u�@A?S�[ƚ?I�5��A
@Y2: 	�v�?rEagerKernelExecute 0*	#��~j�i@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatc �={�?!{��nx:I@)�$xC�?1x��5d�F@:Preprocessing2F
Iterator::Modelz�rK��?!�G�D�@@)l��+֨?1tշ<S�7@:Preprocessing2U
Iterator::Model::ParallelMapV2A��� �?!��'l$@)A��� �?1��'l$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceȘ����?!�$����@)Ș����?1�$����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��7h�?!�)ܫ]�P@)�S�q�?15s����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorv?T1�?!( ȡH@)v?T1�?1( ȡH@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateˡE����?!�����#@)�U����?1����V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapb��A��?!����B~&@)�$�j�d?1���ϯ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t25.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���F�@IV'v=6�O@Q!���Q?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	UގpZ�@UގpZ�@!UގpZ�@      ��!       "	6u�@6u�@!6u�@*      ��!       2	?S�[ƚ??S�[ƚ?!?S�[ƚ?:	�5��A
@�5��A
@!�5��A
@B      ��!       J	2: 	�v�?2: 	�v�?!2: 	�v�?R      ��!       Z	2: 	�v�?2: 	�v�?!2: 	�v�?b      ��!       JGPUY���F�@b qV'v=6�O@y!���Q?@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterm��eU��?!m��eU��?0"1
model/Conv1D_3/conv1dConv2D�j�;��?!��o@H �?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter֙R��?!OĤ�d�?0"1
model/Conv1D_2/conv1dConv2D��آ:��?!�zM�M�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad67�y�?!맞 �?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�xO�?�?!/���^�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�oB ے�?!+�J^��?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterr�Ea�?!��ꆜ�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeHt�;y��?!>)�~���?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��Yʐ_�?!~�5����?Q      Y@Y&W�+�)@a�����U@qd�)s3@y����?"�
both�Your program is MODERATELY input-bound because 5.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t25.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�19.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 