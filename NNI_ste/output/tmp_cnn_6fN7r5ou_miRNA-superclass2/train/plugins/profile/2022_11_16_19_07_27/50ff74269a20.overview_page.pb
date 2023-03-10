�	z�rKC$@z�rKC$@!z�rKC$@	Xs	;�)@Xs	;�)@!Xs	;�)@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLz�rKC$@>"�D@1c���8@A��$��?I�ΤM@Y/�e����?rEagerKernelExecute 0*	�����d@2F
Iterator::Model���QF\�?!Mg�D��C@)��ᔹ��?1���[�;@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Q�|�?!��:@!F5@)�Q�|�?1��:@!F5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�%qVDM�?!�����8@)�fF?N�?1vf�O'5@:Preprocessing2U
Iterator::Model::ParallelMapV2�Z�}�?!|�����'@)�Z�}�?1|�����'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�h;��ʦ?!�Ð�l�;@)�:�/K;�?1lOX5.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zips�<G仸?!��b�N@)�����*�?1~ �x�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��U�P�w?!ځ�yV+@)��U�P�w?1ځ�yV+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa���?!��T9Z=@)R�=�Ne?1��Tv���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 12.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�34.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t28.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Ys	;�)@I�� ŮO@Q*�zX�7@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>"�D@>"�D@!>"�D@      ��!       "	c���8@c���8@!c���8@*      ��!       2	��$��?��$��?!��$��?:	�ΤM@�ΤM@!�ΤM@B      ��!       J	/�e����?/�e����?!/�e����?R      ��!       Z	/�e����?/�e����?!/�e����?b      ��!       JGPUYYs	;�)@b q�� ŮO@y*�zX�7@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput����{s�?!����{s�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����N�?!��m��`�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputP�ɝ���?!oA)��?0"1
model/Conv1D_3/conv1dConv2D�*r�lb�?!�������?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter=��?!�(��`��?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradn��
-�?!QF��D�?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad���	ێ�?!���<�?"1
model/Conv1D_2/conv1dConv2D�g��ʀ�?!7�N�4�?"{
\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�TA��|�?!0&�|ef�?"\
=model/Conv1D_2/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose���q���?!����Y$�?Q      Y@Y&W�+�)@a�����U@q*�	�oB@y�Y7#�?"�
both�Your program is MODERATELY input-bound because 12.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�34.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t28.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�36.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 