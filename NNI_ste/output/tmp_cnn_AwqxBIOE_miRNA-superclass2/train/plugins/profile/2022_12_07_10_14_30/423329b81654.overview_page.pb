�	�iP4H'@�iP4H'@!�iP4H'@	��_	@��_	@!��_	@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�iP4H'@�?��i�@1�߄B�@A�*n�b�?If/�^@Y��Z���?rEagerKernelExecute 0*	f;�O�f@2F
Iterator::Model��lu9�?!��m�lG@)Yni5$�?1�Ie���<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�7�GnM�?!�L��=@)�WV����?1\H�8@:Preprocessing2U
Iterator::Model::ParallelMapV2�߽�Ƅ�?!�u��:2@)�߽�Ƅ�?1�u��:2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��\m���?!��=�L�$@)��\m���?1��=�L�$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateB]�P��?!��)��0@)*Ŏơ~�?1DO��n�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�E�~�?!	s�@�J@)�:��K�?1N��J�e@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorD�.l�V~?!���Ą�@)D�.l�V~?1���Ą�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapY�&�ʠ?!�n3a��2@)�k���f?1���y��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 40.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�37.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��_	@Ins��iS@Q��
�-3@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�?��i�@�?��i�@!�?��i�@      ��!       "	�߄B�@�߄B�@!�߄B�@*      ��!       2	�*n�b�?�*n�b�?!�*n�b�?:	f/�^@f/�^@!f/�^@B      ��!       J	��Z���?��Z���?!��Z���?R      ��!       Z	��Z���?��Z���?!��Z���?b      ��!       JGPUY��_	@b qns��iS@y��
�-3@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd�E���?!d�E���?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad����ʧ?!)~}�7�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad3��O�?!�K/?���?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter%ō�\ͤ?!%��{8#�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��MvES�?!�7f�	��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�:���?!9��)_�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�W�`Y��?!8.�Uǳ�?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�	�E�}�?!wψ|�?"1
model/Conv1D_1/conv1dConv2D6�s��?!86�j5�?"1
model/Conv1D_3/conv1dConv2DdOy�L^�?!$`�*4a�?Q      Y@Y@n]�G*@a8R4��U@q�CU<@y����w��?"�
both�Your program is POTENTIALLY input-bound because 40.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�28.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 