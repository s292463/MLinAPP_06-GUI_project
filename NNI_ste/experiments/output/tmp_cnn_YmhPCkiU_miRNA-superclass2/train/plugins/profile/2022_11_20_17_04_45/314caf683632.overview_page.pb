�	O�\%@O�\%@!O�\%@	�؋���@�؋���@!�؋���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLO�\%@1е/`�?1^����@A
�2��?I'�y�3�?Y/O�R�?rEagerKernelExecute 0*	��C�l3b@2F
Iterator::Model�'�����?!h����F@)���>9
�?1]mܮ�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�p̲'��?!�H�o�<@)#0�70��?1���}	�7@:Preprocessing2U
Iterator::Model::ParallelMapV2Y�n�͓?!,l[��*@)Y�n�͓?1,l[��*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceg��/�?!�n�*�)@)g��/�?1�n�*�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�c�ߛ?!p0�:�2@)Tol�`�?1��U�/O@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�l�M�?!�w]�R<K@)�[�~l�?1�o��,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�׻?~?!�붱�I@)�׻?~?1�붱�I@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�;1��P�?!���DU4@)��A��c?1}�%�P2�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�24.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�؋���@Iu�e6aD@Q���g�L@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	1е/`�?1е/`�?!1е/`�?      ��!       "	^����@^����@!^����@*      ��!       2	
�2��?
�2��?!
�2��?:	'�y�3�?'�y�3�?!'�y�3�?B      ��!       J	/O�R�?/O�R�?!/O�R�?R      ��!       Z	/O�R�?/O�R�?!/O�R�?b      ��!       JGPUY�؋���@b qu�e6aD@y���g�L@�"1
model/Conv1D_2/conv1dConv2DY��l���?!Y��l���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter4;��^ʷ?!F
n5���?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�HmC��?!	v���?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput]P�į?!���7��?0"1
model/Conv1D_3/conv1dConv2DH;����?!�tSr�r�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�JԤd��?!S�����?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad(�Dy��?!M��"��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�%9�
��?!sߥgs.�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradIK��uǘ?!(T����?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<0q�w��?!�3���?0Q      Y@YAd�W�,)@ax��g�U@qbl��RA@y�С琽�?"�
both�Your program is POTENTIALLY input-bound because 16.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�24.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�34.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 