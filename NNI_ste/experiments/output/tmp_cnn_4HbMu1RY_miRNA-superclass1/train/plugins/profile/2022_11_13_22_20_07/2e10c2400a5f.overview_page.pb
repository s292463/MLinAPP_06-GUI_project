�	�T��9 @�T��9 @!�T��9 @	踆��@踆��@!踆��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�T��9 @S�����?1|(ђ�s@A����%n?IN*kg�?Y��Ma��?rEagerKernelExecute 0*	A`��"�b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�[>���?!�ZL]��<@)�\��J�?1��M�7@:Preprocessing2F
Iterator::ModelC�y�'�?!_����9B@)����b)�?1t[ۃ7@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Y-��D�?!���1@)�Y-��D�?1���1@:Preprocessing2U
Iterator::Model::ParallelMapV2��A_z��?!� �U�)@)��A_z��?1� �U�)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���zi��?!�uQ
=�O@)�;��?1�@_�o,!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate������?!4fg(ܑ8@)�d�F ^�?1D�jpA@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���~?!� �n��@)���~?1� �n��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	4��yT�?!	��R:@)/���ިe?1F���K�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�21.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s5.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9鸆��@I�jq��.:@Q�9;�P@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	S�����?S�����?!S�����?      ��!       "	|(ђ�s@|(ђ�s@!|(ђ�s@*      ��!       2	����%n?����%n?!����%n?:	N*kg�?N*kg�?!N*kg�?B      ��!       J	��Ma��?��Ma��?!��Ma��?R      ��!       Z	��Ma��?��Ma��?!��Ma��?b      ��!       JGPUY鸆��@b q�jq��.:@y�9;�P@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterw����?!w����?0"1
model/Conv1D_2/conv1dConv2De^<m>�?!n�n �s�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput5�J/�?!�� Ək�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradW�:%!��?!�1����?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�v\��s�?!d��N*�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradp�j��?�?!KpZKK�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��+��j�?!�)�����?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���,��?!����LD�?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��q�y�?!��<S���?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?����ؘ?!I/��2`�?Q      Y@Y����(@a����c�U@q����;*8@y������?"�
both�Your program is MODERATELY input-bound because 7.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�21.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s5.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�24.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 