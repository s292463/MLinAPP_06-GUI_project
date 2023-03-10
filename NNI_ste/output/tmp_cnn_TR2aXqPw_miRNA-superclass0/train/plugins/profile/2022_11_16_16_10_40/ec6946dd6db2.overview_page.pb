?	???ss@???ss@!???ss@	??m?#@??m?#@!??m?#@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???ss@T?d?C??1qN`:m@A?a?7?W??I???"?@Y?=$|?o??rEagerKernelExecute 0*	j?t?Xf@2F
Iterator::Model?z6?>??!?%A??eI@)?[Ɏ?@??1Sτ???A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!???;@)g?+??2??1B<e?3A8@:Preprocessing2U
Iterator::Model::ParallelMapV2F?v???!}Z?2͏.@)F?v???1}Z?2͏.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipTȕz???!ھ1?H@)L?;?????1#V???u!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??^???!9Äb	?@)??^???19Äb	?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&?(??=??!H <X?&@)?,??;???1W=?M??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?S?<z?!??Lm??@)?S?<z?1??Lm??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapl???D??!?iO%<3)@)-??;??f?1:M?h.???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?38.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??m?#@I??퓈MN@Q??5%<0B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?d?C??T?d?C??!T?d?C??      ??!       "	qN`:m@qN`:m@!qN`:m@*      ??!       2	?a?7?W???a?7?W??!?a?7?W??:	???"?@???"?@!???"?@B      ??!       J	?=$|?o???=$|?o??!?=$|?o??R      ??!       Z	?=$|?o???=$|?o??!?=$|?o??b      ??!       JGPUY??m?#@b q??퓈MN@y??5%<0B@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?낰ZC??!?낰ZC??0"1
model/Conv1D_2/conv1dConv2Dfc:?e??!?ru????"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputG0X?æ?!-Jz?????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad֖p?ä?!???39[??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad[?Ǘ?L??!???????"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterp??D?S??!?bO<??0"1
model/Conv1D_4/conv1dConv2Dԍ???!?????2??"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?????!?ds????"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?X?????!??H??	??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?Ҿ7?ѝ?!?ĥ????Q      Y@Y@n]?G*@a8R4??U@qiVIXrM@@y?'???ڵ?"?
both?Your program is POTENTIALLY input-bound because 21.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?38.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?32.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 