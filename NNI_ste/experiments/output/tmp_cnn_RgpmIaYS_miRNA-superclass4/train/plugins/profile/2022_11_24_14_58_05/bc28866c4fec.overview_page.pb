�	��M�2@��M�2@!��M�2@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��M�2@�-�X @1�^}<�m#@ABv��fG�?I�u7O�@rEagerKernelExecute 0*	2�Z d@2F
Iterator::Model��-s��?!߽!�'MI@)���^a��?19^���(B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+��ص�?!�s:{�:@)�o%;6�?1++�
�:6@:Preprocessing2U
Iterator::Model::ParallelMapV2T�T�	g�?!�~��Ɛ,@)T�T�	g�?1�~��Ɛ,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceGXT��$�?!���	�� @)GXT��$�?1���	�� @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�"���S�?!���.@)��LM��?1�W��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�
�<�?!!B�eزH@)��X����?1��B=�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~�x��{?!*"%��@)�~�x��{?1*"%��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF~�,�?!TT1�11@)7ݲC��f?1�~�<���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 11.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�34.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�ԗ�Q3G@QN+h`��J@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�-�X @�-�X @!�-�X @      ��!       "	�^}<�m#@�^}<�m#@!�^}<�m#@*      ��!       2	Bv��fG�?Bv��fG�?!Bv��fG�?:	�u7O�@�u7O�@!�u7O�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�ԗ�Q3G@yN+h`��J@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�d'0�?!�d'0�?0"1
model/Conv1D_2/conv1dConv2D@�<���?!
���?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterk�πt�?!�l����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�7���?!�[V�Z�?0"1
model/Conv1D_3/conv1dConv2D�@s��¯?!��d��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput��\��+�?!-]��*��?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�-�	qŢ?!sq�b��?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad4|�����?!U�(�v�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad6��{��?!eO�#�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�i��"�?!e8u���?Q      Y@Y.>9\&@a<�?�x4V@q���" hU@y�,�b}��?"�
both�Your program is POTENTIALLY input-bound because 11.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�34.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�85.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 