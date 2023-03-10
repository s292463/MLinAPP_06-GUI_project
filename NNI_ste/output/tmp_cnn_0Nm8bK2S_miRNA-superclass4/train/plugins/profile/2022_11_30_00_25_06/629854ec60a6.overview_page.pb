?	q?q?t@'@q?q?t@'@!q?q?t@'@	8?8???@8?8???@!8?8???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCq?q?t@'@?w??[??1n???("@I??]????Y?[t??z??rEagerKernelExecute 0*	L7?A`?u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??"1A??!??{??K@)???????1??}U!J@:Preprocessing2F
Iterator::Model??U+??!m?\???8@)???n???1^???(?0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?[Ɏ?@??!v?B?}f/@)|?????16w?$?V+@:Preprocessing2U
Iterator::Model::ParallelMapV2T???=??!;?TS?F@)T???=??1;?TS?F@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?mm?y???!6????h@)?mm?y???16????h@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ??q????!?Ԩ??R@)??.o??1+&??'?	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?vٯ;}?!9???> @);?vٯ;}?19???> @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?<֌r??!
&?(?GL@)K????2i?1?b???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?15.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no98?8???@I?؜?j3@Q 7??K?S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?w??[???w??[??!?w??[??      ??!       "	n???("@n???("@!n???("@*      ??!       2      ??!       :	??]??????]????!??]????B      ??!       J	?[t??z???[t??z??!?[t??z??R      ??!       Z	?[t??z???[t??z??!?[t??z??b      ??!       JGPUY8?8???@b q?؜?j3@y 7??K?S@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter(?y(?|??!(?y(?|??0"1
model/Conv1D_2/conv1dConv2DR=???/??!?	WeV??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??]?p???!?s ?=??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?i?(?]??!???8???"3
model/Conv1D_1/BiasAddBiasAdd??a?k/??!D??\&???"1
model/Conv1D_3/conv1dConv2D??\j???!W??????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose???????!??y%???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?:??Ϡ?!T?y(??"-
model/Conv1D_1/ReluRelu.4???!?;74{)??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd????2??!`H?????0Q      Y@Y&W?+?)@a?????U@q?jc=1@y????(??"?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?15.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?17.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 