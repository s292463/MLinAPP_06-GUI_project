�	mt�Oq@mt�Oq@!mt�Oq@	���m��@���m��@!���m��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLmt�Oq@'�+�V]�?1��%��@A���'�T?I�,�i��@Y�6�����?rEagerKernelExecute 0*	�C�l�3d@2F
Iterator::Model�D�$�?! G�I@)ᶶ�T�?1�ҫcKA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��=�W�?!;�ޒ�j=@)���i�:�?1L����9@:Preprocessing2U
Iterator::Model::ParallelMapV2��z2��?!����1�0@)��z2��?1����1�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>u�Rz��?!���i��@)>u�Rz��?1���i��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ����?!hu���{)@)��F����?1���Vc@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�dV�p;�?!��߸sH@)����=�?1�=�˃t@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�d�x?!sw��@)�d�x?1sw��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB^&�Ǘ?!|���Ǽ,@)���i�e?1�زh��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���m��@IQ�k?��C@Q�N�Y��L@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	'�+�V]�?'�+�V]�?!'�+�V]�?      ��!       "	��%��@��%��@!��%��@*      ��!       2	���'�T?���'�T?!���'�T?:	�,�i��@�,�i��@!�,�i��@B      ��!       J	�6�����?�6�����?!�6�����?R      ��!       Z	�6�����?�6�����?!�6�����?b      ��!       JGPUY���m��@b qQ�k?��C@y�N�Y��L@�"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad}q���?!}q���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterC�7�{�?!�*�ٽ�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��
xR��?!���=na�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterf4���?!�����?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose��N��ժ?!R0��=��?"3
model/Conv1D_1/BiasAddBiasAdd�U�_i�?!��|i5�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose*�&]T�?!��� ��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	TransposeDzUQL�?!��K��?"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose>��̩?!_q9���?"-
model/Conv1D_1/ReluReluJE�le�?!�)�ׇ�?Q      Y@Yyxxxxx*@a�����U@qK��v��9@yѻ_���?"�
device�Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�25.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 