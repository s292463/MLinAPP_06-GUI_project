?	??? ??@??? ??@!??? ??@	6;UT????6;UT????!6;UT????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??? ??@???h o??1\ ?k\?@A?5Φ#?k?IR'????E@Y?c\qqT??rEagerKernelExecute 0*	??ʡMa@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateMK??F>??!???"?e@@)]~p>??1??da?=@:Preprocessing2F
Iterator::ModelϽ?K?;??!??zx?D@)??f????1???;6@:Preprocessing2U
Iterator::Model::ParallelMapV2?1^???!?n W3@)?1^???1?n W3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???'?.??!??ʁ??3@)?[??.???1~???/@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-?x?!ˋ??@)??-?x?1ˋ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip]?&?Ҵ?!?w??`M@)?2p@Kw?1U?6??n@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`?o`r???!?????aA@)46<?Rf?1m?>?9??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorٓ??<c?!~ {c?#??)ٓ??<c?1~ {c?#??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlices???Y?!>?sn"???)s???Y?1>?sn"???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no96;UT????I i?ȶ8@QT??W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???h o?????h o??!???h o??      ??!       "	\ ?k\?@\ ?k\?@!\ ?k\?@*      ??!       2	?5Φ#?k??5Φ#?k?!?5Φ#?k?:	R'????E@R'????E@!R'????E@B      ??!       J	?c\qqT???c\qqT??!?c\qqT??R      ??!       Z	?c\qqT???c\qqT??!?c\qqT??b      ??!       JGPUY6;UT????b q i?ȶ8@yT??W@?"1
model/Conv1D_2/conv1dConv2D?Z"?T??!?Z"?T??"1
model/Conv1D_3/conv1dConv2D?Q???I??!??/r|??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?	g?v??!%ē?&??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?<????!EӃ?,	??0"1
model/Conv1D_4/conv1dConv2D?6]_
[??!H$?n??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputa?E9܃?!?_?l	???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?_ʸs8??!8??;????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterr?s\???!Qp???=??0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter	?QJ?~?!ӗ$#;{??0"1
model/Conv1D_1/conv1dConv2D??hYs?{?!3i?	????Q      Y@Y??\??@a6zWd?W@q?0?U?B@y??Z??I?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?37.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 