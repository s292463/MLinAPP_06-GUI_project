?	??_=??@??_=??@!??_=??@	Snq?@Snq?@!Snq?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??_=??@V??#))??1E7?@AH4?"1??I?G??
k@YrR??8???rEagerKernelExecute 0*	?????u@2U
Iterator::Model::ParallelMapV2 ???Z??!>ܽȏ?I@) ???Z??1>ܽȏ?I@:Preprocessing2F
Iterator::Model?ګ????!70??
?Q@)???Y??1[)?2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatG?&ji??!???)@)???v?Ӣ?1???S?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??M+?@??!???h?)@)???!6X??1,??[q-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??4?(??!'&WV`U@)??4?(??1'&WV`U@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?b??V??!'?[\?S=@)n?ݳ???1???hw@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-?|?!?Vͣ?? @)??-?|?1?Vͣ?? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???ا?!?i=^?+@)?-X?xi?1???Q?z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Snq?@I)?`=?N@Q?????xA@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V??#))??V??#))??!V??#))??      ??!       "	E7?@E7?@!E7?@*      ??!       2	H4?"1??H4?"1??!H4?"1??:	?G??
k@?G??
k@!?G??
k@B      ??!       J	rR??8???rR??8???!rR??8???R      ??!       Z	rR??8???rR??8???!rR??8???b      ??!       JGPUYSnq?@b q)?`=?N@y?????xA@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??Y?޽?!??Y?޽?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?ĊU??!ff*????0"1
model/Conv1D_2/conv1dConv2DvbN?I$??!?˨?V??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???????!f?ňr???0"1
model/Conv1D_3/conv1dConv2D?B?|??!??????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???*ɿ??!--?c???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput7+oU=??!?9֧??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?:?׺О?!????????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?q/zܚ?!\??B??0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposeX???>:??!	???&k??Q      Y@Ym۶m۶)@a?$I?$?U@q?Ejy?T=@y?8??L???"?
both?Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?41.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?29.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 