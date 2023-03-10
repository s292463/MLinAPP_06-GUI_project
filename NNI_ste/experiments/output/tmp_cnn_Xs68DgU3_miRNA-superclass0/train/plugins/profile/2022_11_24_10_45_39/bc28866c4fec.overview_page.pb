?	l???D @l???D @!l???D @	Dx??W?@Dx??W?@!Dx??W?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLl???D @g{??????1???6?@A?EИIԛ?I?N[#?@Y?????Q??rEagerKernelExecute 0*9??v?'k@)       =2F
Iterator::Model$C??g??!??@51?D@)??a?1??1????n?>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??
a5???!OsjU&?=@)???/JЧ?1?N?h5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatY?????!қ1c?2@):???`???1?=y?-@:Preprocessing2U
Iterator::Model::ParallelMapV2AH0?[??!??HC??$@)AH0?[??1??HC??$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???#bJ??!h???JM@)ro~?D???1?h??-q"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~?$A???!f???? @)~?$A???1f???? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_???|?!f??!ީ	@)??_???|?1f??!ީ	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??E
e??! DG?@)?Z(???i?1??(?~>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?36.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Dx??W?@IG0T?`M@Q?m1?MB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	g{??????g{??????!g{??????      ??!       "	???6?@???6?@!???6?@*      ??!       2	?EИIԛ??EИIԛ?!?EИIԛ?:	?N[#?@?N[#?@!?N[#?@B      ??!       J	?????Q???????Q??!?????Q??R      ??!       Z	?????Q???????Q??!?????Q??b      ??!       JGPUYDx??W?@b qG0T?`M@y?m1?MB@?"1
model/Conv1D_2/conv1dConv2D?լ??N??!?լ??N??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter4iM????!j}?٭??0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad}?2t?@??!?Ή?	>??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradi???e??!?I??qW??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?ǲh????!??Y?????0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?b?u?_??!$JިK??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?1??!???a????"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?5?ļ??!;???x???"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeP:L*?.??!?YyV???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	TransposeyOe????!t?????Q      Y@Ym۶m۶)@a?$I?$?U@q?r#??>@yS??m???"?
both?Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?36.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 