?	m?s??@m?s??@!m?s??@	?gyQ@?gyQ@!?gyQ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLm?s??@??8?j???1??xy:???A??>s֧??I}?;l"?@YT? ?!???rEagerKernelExecute 0*	{?G?ra@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatȗP????!f????>@)/ܹ0ҋ??1?@???9@:Preprocessing2F
Iterator::Model^??v1??!)??blD@)e?fb??1:??"?9@:Preprocessing2U
Iterator::Model::ParallelMapV2??W:???!0??`E?.@)??W:???10??`E?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???y7??!5?
?,@)???y7??15?
?,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN??????!e?g3?4@)/?
ҌE??1?k??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipJ??Gp#??!???/??M@)?ݮ????1H?o,o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?^??x?z?!^39?@?@)?^??x?z?1^39?@?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????b??!)?Z>??6@)j?drjgh?1?葴?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?46.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?gyQ@IS??l?R@Q??d,;9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??8?j?????8?j???!??8?j???      ??!       "	??xy:?????xy:???!??xy:???*      ??!       2	??>s֧????>s֧??!??>s֧??:	}?;l"?@}?;l"?@!}?;l"?@B      ??!       J	T? ?!???T? ?!???!T? ?!???R      ??!       Z	T? ?!???T? ?!???!T? ?!???b      ??!       JGPUY?gyQ@b qS??l?R@y??d,;9@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???5??!???5??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??Q? ??!?RM?????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad6O?`%???!???L?D??"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsu?7{??!?S?????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??O1y??!??3?t??0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?i??Π?!???Q-??"1
model/Conv1D_1/conv1dConv2Db ?üJ??!([Um?6??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputl?q}{p??!v,%???0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???e???!?w}????0"1
model/Conv1D_2/conv1dConv2D?}?F?ٜ?!??㡄???Q      Y@Y?ܺ?+@a?p?h?U@qq?????B@y??g???"?
both?Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?46.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?37.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 