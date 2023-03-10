�	4e���@4e���@!4e���@	o���� @o���� @!o���� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL4e���@k}�Ж��?1m���E@A�����%�?IGY����@Y�~j�t��?rEagerKernelExecute 0*	�Vqq@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���v��?!fsTwG@)Ve����?1�*\��E@:Preprocessing2F
Iterator::Model]�gA(�?!�b)E�@@)<L����?1xB��9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��Ϸ�?!Ľ�m�*@){����?1_�-�T&@:Preprocessing2U
Iterator::Model::ParallelMapV2�t�i��?!���U�� @)�t�i��?1���U�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipH�'���?!��skݟP@).���=��?1śB+9h@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�MbX9�?!���O@)�MbX9�?1���O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor F�6�x?!�?�/@) F�6�x?1�?�/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���V	�?!��O;��G@)Mۿ�Ҥd?1�eY�y��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 27.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�37.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9o���� @I����@P@Q��H��r@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	k}�Ж��?k}�Ж��?!k}�Ж��?      ��!       "	m���E@m���E@!m���E@*      ��!       2	�����%�?�����%�?!�����%�?:	GY����@GY����@!GY����@B      ��!       J	�~j�t��?�~j�t��?!�~j�t��?R      ��!       Z	�~j�t��?�~j�t��?!�~j�t��?b      ��!       JGPUYo���� @b q����@P@y��H��r@@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterb��ɟǮ?!b��ɟǮ?0"1
model/Conv1D_2/conv1dConv2D��~��r�?!ʷOO�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�H�7��?!$�^�u�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad!2���?!l7���Q�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterUô1Ϟ?!�����+�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputC�a2���?!?.����?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterh��<$��?!���#��?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposem����?!���ԕ�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�5��Rқ?!�H�@�R�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeZaO���?!�^�%Y�?Q      Y@Yyxxxxx*@a�����U@q6N"�C@yT3���+�?"�
both�Your program is POTENTIALLY input-bound because 27.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�39.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 