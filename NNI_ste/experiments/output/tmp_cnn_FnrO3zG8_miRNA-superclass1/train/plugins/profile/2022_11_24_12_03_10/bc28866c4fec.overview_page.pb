� 	-�LN�@-�LN�@!-�LN�@	/��¯@/��¯@!/��¯@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL-�LN�@��U��B�?1�gz��L@A�-u�׃�?IR����@YLm����?rEagerKernelExecute 0*	���ƀ�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap� O!W�?!�_��I@);ŪA�?1�㊎`�G@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map���/J��?!�_5�ZB@)Ĵo��?1�Ǻ��\@@:Preprocessing2F
Iterator::Model(�H0�̲?!bb����@)Ral!�A�?1\u/�@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat/�
Ҍ�?!�|��h�@)e����`�?1A����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����
�?!
4fOQ+L@)��l���?1��W�	@:Preprocessing2U
Iterator::Model::ParallelMapV2��I~į�?!��ҡB@)��I~į�?1��ҡB@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate%y���A�?!�e�~8�?)����*�?1b���e�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatI�\߇��?!POd��X�?)�k��F�?1?4 � a�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch`���f�?!�ȗǨt�?)`���f�?1�ȗǨt�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�/K;5�{?!�2\�h�?)�/K;5�{?1�2\�h�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV�F�?x?!#6�,���?)V�F�?x?1#6�,���?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeuʣaq?!�S�޵�?)uʣaq?1�S�޵�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate����@�?!�����?)$c���a?1x�ab�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��L�nQ?!�u�	ʹ?)��L�nQ?1�u�	ʹ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�39.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9/��¯@IpwɧL@Q�/�>�B@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��U��B�?��U��B�?!��U��B�?      ��!       "	�gz��L@�gz��L@!�gz��L@*      ��!       2	�-u�׃�?�-u�׃�?!�-u�׃�?:	R����@R����@!R����@B      ��!       J	Lm����?Lm����?!Lm����?R      ��!       Z	Lm����?Lm����?!Lm����?b      ��!       JGPUY/��¯@b qpwɧL@y�/�>�B@�"1
model/Conv1D_2/conv1dConv2Dʌv�2�?!ʌv�2�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��މ��?!F?��hi�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput���԰?!�r�t��?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad���?59�?!�w��0�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�1ڌR�?!����2;�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradI7�/ٛ�?!�<��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�ƌ9��?!rۉ@���?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��&��?!E|?lr�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposeY{R��0�?!�l!w�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose4�?!������?Q      Y@Y��8+?!4@a��15��S@q7u�eox*@y6"����?"�
both�Your program is POTENTIALLY input-bound because 17.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�39.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�13.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 