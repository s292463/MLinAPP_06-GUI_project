?	?? 4Jg!@?? 4Jg!@!?? 4Jg!@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?? 4Jg!@??B???1?l?M@A?$????I]~p?@rEagerKernelExecute 0*	P??n?a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU?Y??!1?w7??;@)??.????1?emF%7@:Preprocessing2U
Iterator::Model::ParallelMapV2?~?f+??!??R7?5@)?~?f+??1??R7?5@:Preprocessing2F
Iterator::ModelM?^?iN??!??+|OE@)?ť*mq??1?bn?g?4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???.5B??!\???5@)̳?V|C??1bg??(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceI?F?q???!V???V~#@)I?F?q???1V???V~#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?	Q???!|Rԃ??L@)??T?-???1y?S?x?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9Q???{?!???)?U@)9Q???{?1???)?U@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ߠ????!i㛮??7@)??N?0?e?1̠.????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIl???p$I@Q?&E??H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??B?????B???!??B???      ??!       "	?l?M@?l?M@!?l?M@*      ??!       2	?$?????$????!?$????:	]~p?@]~p?@!]~p?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb ql???p$I@y?&E??H@?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?h?K?X??!?h?K?X??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter|?W??N??!?*??S??0"1
model/Conv1D_2/conv1dConv2Dg^??p??!8?ր??"1
model/Conv1D_3/conv1dConv2D?&'?ҍ??!۷???~??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter1?m?????!ԕ??u???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputQB=??<??!>????0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?Θ?????!?W?B?l??0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits??=?Rƞ?!?4?nY??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad4??tq$??!?o??V;??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??g ???!???5???Q      Y@Y?????(@a????c?U@q׈?`?RJ@y???df???"?
both?Your program is POTENTIALLY input-bound because 8.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?41.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?52.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 