?	|,}??:!@|,}??:!@!|,}??:!@	߼?t@߼?t@!߼?t@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL|,}??:!@??o??R??1?a?'?i??AޫV&?R??I?????_@Ym???{???rEagerKernelExecute 0*	+??v@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW??U???!x?lA0O@)x??Dg???1\?v???M@:Preprocessing2F
Iterator::Modele8?πz??!t????4@)???d???1?~?%-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;R}?%??!???*@)???/??1???P??%@:Preprocessing2U
Iterator::Model::ParallelMapV2^?}t?ʗ?!η(???@)^?}t?ʗ?1η(???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????㾅?!?Qf??k@)????㾅?1?Qf??k@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd?w?W??!c?M:^?S@)?t{Ic??1,[|(=?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?
E??S??!b?)??@)?
E??S??1b?)??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Xm?_??!j?K+!?O@)?%??:?j?1?|??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?59.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9߼?t@I?3+???S@Q?Y??)3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??o??R????o??R??!??o??R??      ??!       "	?a?'?i???a?'?i??!?a?'?i??*      ??!       2	ޫV&?R??ޫV&?R??!ޫV&?R??:	?????_@?????_@!?????_@B      ??!       J	m???{???m???{???!m???{???R      ??!       Z	m???{???m???{???!m???{???b      ??!       JGPUY߼?t@b q?3+???S@y?Y??)3@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputV?????!V?????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?R?O?B??!????t??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??]C????!6ڴ?V??0"1
model/Conv1D_2/conv1dConv2D????֬?!????`???"1
model/Conv1D_3/conv1dConv2DjZ"*????!G???wa??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput܋)P?:??!??Ш??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??Q,??!?'?6???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad< '
?M??!?|?????"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad???~?՛?!???
r\??"1
model/Conv1D_1/conv1dConv2Dq
?????!R??????Q      Y@Y&W?+?)@a?????U@q?*?>@y?Z??N???"?
both?Your program is POTENTIALLY input-bound because 18.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?59.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 