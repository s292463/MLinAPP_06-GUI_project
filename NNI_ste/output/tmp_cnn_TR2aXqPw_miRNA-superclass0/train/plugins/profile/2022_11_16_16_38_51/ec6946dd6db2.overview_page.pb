?	??Xǁ@??Xǁ@!??Xǁ@	, A]?@, A]?@!, A]?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??Xǁ@??%VFc??1_^?}tJ	@A9??????I+n?b~n@Y?L??O??rEagerKernelExecute 0*	? ?rhb@2F
Iterator::Model%\?#????!?[???I@)?-?????1?h?K?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?؀q??!????<@)Uh ????1c??I/i8@:Preprocessing2U
Iterator::Model::ParallelMapV2y??n?U??!?b?.@)y??n?U??1?b?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicem?i?*???!.?a@??!@)m?i?*???1.?a@??!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate~?D?<??!?Rf?,@)??~?{??1?y??o7@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipjO?9????!Z?>=>?H@)?????}?1ib<?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ʼUסz?!?r??R?@)?ʼUסz?1?r??R?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?A{??З?!?%??0@)
?????d?1?A??^???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?33.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9, A]?@I6????JI@Q??$?a?F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??%VFc????%VFc??!??%VFc??      ??!       "	_^?}tJ	@_^?}tJ	@!_^?}tJ	@*      ??!       2	9??????9??????!9??????:	+n?b~n@+n?b~n@!+n?b~n@B      ??!       J	?L??O???L??O??!?L??O??R      ??!       Z	?L??O???L??O??!?L??O??b      ??!       JGPUY, A]?@b q6????JI@y??$?a?F@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??V???!??V???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter]i<??I??!??`????0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputF ?????!?@e?p??0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??ۈɲ??!yzw?E???0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput???U??!ytED???0"1
model/Conv1D_4/conv1dConv2D??\?
??!I ??[??"1
model/Conv1D_3/conv1dConv2D?W?g:z??!E??,??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?p?????!\???????"1
model/Conv1D_2/conv1dConv2Db?J? ??!??( ??"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad!?X?O???!0?\!%???Q      Y@Y??u@7?)@a%D?9?U@q?q!??A@y????\??"?
both?Your program is POTENTIALLY input-bound because 16.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?33.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?35.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 