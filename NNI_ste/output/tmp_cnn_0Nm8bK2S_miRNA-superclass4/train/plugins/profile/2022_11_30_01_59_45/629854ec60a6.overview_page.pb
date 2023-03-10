?	s?,&6?1@s?,&6?1@!s?,&6?1@	|??0??|??0??!|??0??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCs?,&6?1@Pmp"?5??1{?G??.@I9ӄ?'???Y??w??x??rEagerKernelExecute 0*	??S??Cf@2F
Iterator::Model_	?Į???!??U??H@)?o????1??^?a@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ݓ????!??+?=@)5'/2???1?>??D?8@:Preprocessing2U
Iterator::Model::ParallelMapV2R)v4???!??"?.@)R)v4???1??"?.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipi?QH2???!s?D?I@)I???p???1q$???)"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?D??)??!?????~@)?D??)??1?????~@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateSh?
??!????+(@)
?F???1???.??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor
?s34~?!^?:?g?@)
?s34~?1^?:?g?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapoӟ?H??!?kup?|+@)?h???2h?1m?+???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9|??0??I?=l??'@QCy,?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Pmp"?5??Pmp"?5??!Pmp"?5??      ??!       "	{?G??.@{?G??.@!{?G??.@*      ??!       2      ??!       :	9ӄ?'???9ӄ?'???!9ӄ?'???B      ??!       J	??w??x????w??x??!??w??x??R      ??!       Z	??w??x????w??x??!??w??x??b      ??!       JGPUY|??0??b q?=l??'@yCy,?U@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter|?lǷ?!|?lǷ?0"1
model/Conv1D_2/conv1dConv2D??:?Դ?!s??]?M??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput5????N??!??Y?a??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFiltere '?ܧ?!??Q?Y???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??y??Z??!L??ۯ7??"1
model/Conv1D_3/conv1dConv2D??????!?ߑ+????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradH??Fι??!L?b?????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?u%7?o??!?
?g????"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	TransposemT&>|S??!?o?+????"3
model/Conv1D_1/BiasAddBiasAdd???e{w??!-?iv??Q      Y@Ym۶m۶)@a?$I?$?U@qs????24@yr?0)??"?
both?Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 