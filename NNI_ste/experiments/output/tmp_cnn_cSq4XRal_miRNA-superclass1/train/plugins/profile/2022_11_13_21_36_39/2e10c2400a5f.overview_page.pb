�	�:��k@�:��k@!�:��k@	��ծ�@��ծ�@!��ծ�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�:��k@������?1-
�(z`�?A�;O<g�?I��,@Y�l����?rEagerKernelExecute 0*	�n��j`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��>+�?!��4�{@@)� �س�?1���8<@:Preprocessing2F
Iterator::Model��'��?!N�93��B@)�����̜?1��Gj5@:Preprocessing2U
Iterator::Model::ParallelMapV2��j�=�?!�;�w�0@)��j�=�?1�;�w�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��|y��?!��Cʵ*@)��|y��?1��Cʵ*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�g�����?!JI$���4@)wH1@�	�?1��&�b�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipǸ��ܴ?!�i��4O@)6����$�?1�`߫�~@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,=)�z?!a�s��h@),=)�z?1a�s��h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapk*��.��?!# ��J�6@){�G�zd?1�m���t�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�56.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��ծ�@I!8���Q@Q'm j��:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "	-
�(z`�?-
�(z`�?!-
�(z`�?*      ��!       2	�;O<g�?�;O<g�?!�;O<g�?:	��,@��,@!��,@B      ��!       J	�l����?�l����?!�l����?R      ��!       Z	�l����?�l����?!�l����?b      ��!       JGPUY��ծ�@b q!8���Q@y'm j��:@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterF9�^��?!F9�^��?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�kN/뙤?!2Ԧ���?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad��^���?!ⅾS��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�QK��M�?!XZ���?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���Π?!%j��h��?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�2k2S�?!}��GS��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��`.�?!��25��?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad=3!l�?!����E��?"1
model/Conv1D_2/conv1dConv2DW[j���?!��K���?"3
model/Conv1D_1/BiasAddBiasAdd�^�`���?!�V�	o�?Q      Y@YVUUUUU)@aUUUUU�U@qh�M�o�B@y���9�Q�?"�
both�Your program is POTENTIALLY input-bound because 13.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�56.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�37.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 