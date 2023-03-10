?	'?ҨP@'?ҨP@!'?ҨP@	jF?-?a@jF?-?a@!jF?-?a@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL'?ҨP@!"5?b??1{?ۡa?@A?Q*?	???I6?
?r@Y?D????rEagerKernelExecute 0*	I+?Nc@2F
Iterator::Model???[???!$??Q}G@)?tp?x??13???@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!?K????!r??du?@)?\??7??1??d??:@:Preprocessing2U
Iterator::Model::ParallelMapV2?M?g\??!??????-@)?M?g\??1??????-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??+??ؕ?!?7#??+@)?Nw?xΆ?1D?T???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)z?c????!??fi@))z?c????1??fi@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?uŌ????!?y5???J@)???o??1?'?|O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?p;4,F}?!g?stR?@)?p;4,F}?1g?stR?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapK\Ǹ????!?#?T5x/@)?ۼqRh?1?`?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?39.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9jF?-?a@I? ??@J@Q{?iF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!"5?b??!"5?b??!!"5?b??      ??!       "	{?ۡa?@{?ۡa?@!{?ۡa?@*      ??!       2	?Q*?	????Q*?	???!?Q*?	???:	6?
?r@6?
?r@!6?
?r@B      ??!       J	?D?????D????!?D????R      ??!       Z	?D?????D????!?D????b      ??!       JGPUYjF?-?a@b q? ??@J@y{?iF@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??	p???!??	p???0"1
model/Conv1D_4/conv1dConv2D?ѿ???!????@)??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?>?M?C??!'??E????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??yӻ???!$?:???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputt?Β"???!7h?o???0"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam?g?R묦?!5U?&???"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterRR{?ܥ?!??o?????0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?<?????!?f?????0"1
model/Conv1D_3/conv1dConv2D????Þ?!???%????"1
model/Conv1D_2/conv1dConv2D?p??%???!??S?ix??Q      Y@Y?Cc}(@a????S?U@q?=#?>@y??Vr????"?
both?Your program is POTENTIALLY input-bound because 12.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?39.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 