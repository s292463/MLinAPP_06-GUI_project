?	????mЁ@????mЁ@!????mЁ@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????mЁ@?jׄ????1`??VO?@IS???"/L@rEagerKernelExecute 0*	?l????t@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?R	O????!V?????Q@)CX?%????14d?Q@:Preprocessing2F
Iterator::Model><K?P??!x?] ??4@)2?g?o}??1 W??-@:Preprocessing2U
Iterator::Model::ParallelMapV2??]gE??!?g??3@)??]gE??1?g??3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatМ?)?d??!]??6?@)$??P??1?m?U-Y	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???dȱ??!???Z?S@)D0.s~?1??@??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor+?m??z?!`LL?? @)+?m??z?1`LL?? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Za?^C??!>"???Q@)??֪]c?1?s9ן??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor?1??l^?!Q?5??)?1??l^?1Q?5??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice&:?,B?U?!?3?????)&:?,B?U?1?3?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?	????#@Q?>H???V@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?jׄ?????jׄ????!?jׄ????      ??!       "	`??VO?@`??VO?@!`??VO?@*      ??!       2      ??!       :	S???"/L@S???"/L@!S???"/L@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?	????#@y?>H???V@?"1
model/Conv1D_2/conv1dConv2D????#M??!????#M??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter-???D"??!v???7??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??KW???!?Ĵ?_???0"1
model/Conv1D_3/conv1dConv2D	S?I????!R?E??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?'?M;??!?[?l???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?R? ߼??!<?a`=??0"1
model/Conv1D_4/conv1dConv2D?9sZ??!#vܢȲ??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput^hP
l???!ķS~??0"1
model/Conv1D_1/conv1dConv2DαJ????!?⺢?M??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter]8?&????!l?V?????0Q      Y@Y?{?1m@aD?,??W@q??(?"?@yY"?%?+I?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 