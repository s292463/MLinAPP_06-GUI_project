?	`>Y1\?@`>Y1\?@!`>Y1\?@	{?⽣@{?⽣@!{?⽣@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL`>Y1\?@??y ?|??1l]j?~?@AM???$??I?O ???@Y<0?????rEagerKernelExecute 0*	|?5^??r@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?\6:????!?l???O@)'?????1_?o??N@:Preprocessing2F
Iterator::ModelǺ?????!b?)?4@)0??乾??1׋????$@:Preprocessing2U
Iterator::Model::ParallelMapV2^??6S!??!???#@)^??6S!??1???#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~p>u?R??!$6j;c!@)~p>u?R??1$6j;c!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateD??~?Ϣ?!(?"?b(@)^????1?ǃ??L@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip>z?}????!'????S@){Cr2??1??o>?J@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?f+/??!?W????@)?f+/??1?W????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph?o}Xo??!?O`??|*@)4?????i?1?=?n>???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?46.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t10.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9{?⽣@I??s?ԿL@QHy??A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??y ?|????y ?|??!??y ?|??      ??!       "	l]j?~?@l]j?~?@!l]j?~?@*      ??!       2	M???$??M???$??!M???$??:	?O ???@?O ???@!?O ???@B      ??!       J	<0?????<0?????!<0?????R      ??!       Z	<0?????<0?????!<0?????b      ??!       JGPUY{?⽣@b q??s?ԿL@yHy??A@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??@;?6??!??@;?6??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??۞???!??Ǩ???"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?0?p,ߦ?!$'?p???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?xB~?h??!Q?*?B??0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradF?`k? ??!??i$??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?tM#?
??!
1?y?S??0"1
model/Conv1D_2/conv1dConv2Dd??N?Q??!??ǃ?]??"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?5s?????!????g]??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?B?9ן?!(8??Z??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose-???
Þ?!,TA8	G??Q      Y@Y      )@a     ?U@q?+H???@y?u??@??"?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?46.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t10.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 