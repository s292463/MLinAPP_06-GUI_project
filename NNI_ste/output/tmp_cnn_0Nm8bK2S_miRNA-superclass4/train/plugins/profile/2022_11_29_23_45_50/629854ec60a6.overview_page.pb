?	?v??C@?v??C@!?v??C@	?ٔuv@?ٔuv@!?ٔuv@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?v??C@??4?ׂ??1??ʆ5?@I??W???Y?$A?
??rEagerKernelExecute 0*	43333?e@2F
Iterator::Model?ᔹ?F??!?????F@)sI?v??1??3?C??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??6?[??!?T??P?;@)[^??6S??1G??ؚ3@:Preprocessing2U
Iterator::Model::ParallelMapV2o??;????!???w?=,@)o??;????1???w?=,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?A??????!VC+?K@)?@?ش??1wRVC+%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@??$"??!?c???@)@??$"??1?c???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?0?????!&??s?}@)?0?????1&??s?}@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???9???!?@???1,@)?&??d???1?FlM?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapsg&?5??!e~	 ?/@)??E??\j?1l?̰???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?29.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?ٔuv@I?Q?yɯ>@Qݻ?4XO@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??4?ׂ????4?ׂ??!??4?ׂ??      ??!       "	??ʆ5?@??ʆ5?@!??ʆ5?@*      ??!       2      ??!       :	??W?????W???!??W???B      ??!       J	?$A?
???$A?
??!?$A?
??R      ??!       Z	?$A?
???$A?
??!?$A?
??b      ??!       JGPUY?ٔuv@b q?Q?yɯ>@yݻ?4XO@?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter{??Cza??!{??Cza??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter L?H????!?t????0"1
model/Conv1D_3/conv1dConv2D?F???!K??:s??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput%\?G????!T??ל???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterZfg?:՜?!?ZɅ?|??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad'????!+}???.??"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad??`?????!??:k]???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??U?U??!cJ]???0"1
model/Conv1D_2/conv1dConv2D???خ??!?p???L??"{
\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposemA?3)ޔ?!?jT????Q      Y@Yyxxxxx*@a??????U@q????;?6@yVNK ?ר?"?
both?Your program is MODERATELY input-bound because 7.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?29.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 