?	ެ??J @ެ??J @!ެ??J @	`?????@`?????@!`?????@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLެ??J @?~?T?B??1???S@A[?[!?Ƣ?I?fH?k??Y??
~??rEagerKernelExecute 0*	P??n?s@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@4?????!?߃??N@)?W}w??1ځL???L@:Preprocessing2F
Iterator::Model???#*T??!??6?Z3@)?'?H0դ?1?fv??)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Ov3???!???;?3-@)?:9Cqǣ?1??Q5fo(@:Preprocessing2U
Iterator::Model::ParallelMapV2ˡE?????!??@)ˡE?????1??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??=]ݱ??!??0I?@)??=]ݱ??1??0I?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?pX?Q??!C2L{)T@)?߆?y??1>???܇
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????~?!?w? @)????~?1?w? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapO??e?c??!???9?]O@)??^(`;h?1D??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?21.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t16.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9`?????@I?#?;?B@Q???L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~?T?B???~?T?B??!?~?T?B??      ??!       "	???S@???S@!???S@*      ??!       2	[?[!?Ƣ?[?[!?Ƣ?![?[!?Ƣ?:	?fH?k???fH?k??!?fH?k??B      ??!       J	??
~????
~??!??
~??R      ??!       Z	??
~????
~??!??
~??b      ??!       JGPUY`?????@b q?#?;?B@y???L@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??;t???!??;t???0"1
model/Conv1D_2/conv1dConv2DNRd????!i&??????"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<??8mۮ?!????q??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput%??2????!A???I??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad06?5?s??!??+8??"1
model/Conv1D_3/conv1dConv2D?*I񸘤?!XB??B???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterl?F???!<ڕk!??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad_???')??!2???f??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradT??C?3??!?a???)??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputzɕ???!????S???0Q      Y@Y@n]?G*@a8R4??U@q?=u??6@y?_k????"?
both?Your program is MODERATELY input-bound because 6.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?21.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t16.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 