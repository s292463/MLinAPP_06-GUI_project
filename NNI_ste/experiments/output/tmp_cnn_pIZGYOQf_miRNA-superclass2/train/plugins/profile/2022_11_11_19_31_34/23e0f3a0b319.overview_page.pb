�	/�KR��@/�KR��@!/�KR��@	%���O�?%���O�?!%���O�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL/�KR��@VҊo(<�?14�s�k@As0� ��?I��w��8
@Y���M+�?rEagerKernelExecute 0*	�~j�tgg@2F
Iterator::Model�ο]��?!g��x]L@)�h:;�?1����KF@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��Q�ª?!�R9]�;@)�\6:秨?1QtF�9@:Preprocessing2U
Iterator::Model::ParallelMapV2��t?�?!T���E@(@)��t?�?1T���E@(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatIIC��?!�M�M��$@)6\䞮�?10��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!�@S���E@)�&p�n~?1.��E�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorl\���|?!�'��@)l\���|?1�'��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+�MF�a�?!LE��=@)='�o|�i?1��c[��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?RD�U�a?!�]ZTB��?)?RD�U�a?1�]ZTB��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_?!~Na *��?)���_?1~Na *��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9%���O�?Iդ�R�sP@Ql���'@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	VҊo(<�?VҊo(<�?!VҊo(<�?      ��!       "	4�s�k@4�s�k@!4�s�k@*      ��!       2	s0� ��?s0� ��?!s0� ��?:	��w��8
@��w��8
@!��w��8
@B      ��!       J	���M+�?���M+�?!���M+�?R      ��!       Z	���M+�?���M+�?!���M+�?b      ��!       JGPUY%���O�?b qդ�R�sP@yl���'@@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterPA%�*ƭ?!PA%�*ƭ?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradL���}�?!N��dTZ�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradؚ���?!]��,��?"1
model/Conv1D_2/conv1dConv2D۵��A�?!�H�{�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterh4i�X��?!�ѭ�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposef���Ρ?!��M2��?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�Q?���?!�8Z�B�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�=]�-[�?!���n�?"3
model/Conv1D_1/BiasAddBiasAdd�:�`Ġ?!�u�����?"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�aׄ��?!��@,��?Q      Y@Y�ґ=Q)@a��M���U@q����`H@yy��Z6��?"�
both�Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�46.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�48.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 