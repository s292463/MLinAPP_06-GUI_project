? 	?˵h?"@?˵h?"@!?˵h?"@	B&??!~??B&??!~??!B&??!~??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?˵h?"@>?$@MM@1??9z| @A?-=??ɜ?I?(5J@Yg~5??rEagerKernelExecute 0*?p=
ʧ@)      `=2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?????@!bYq}?xT@)??? ?@1??c?T@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ơ~???!?????}.@)?=?$@M??1P+?,@:Preprocessing2F
Iterator::Model?Բ??H??!?1??4? @)h?4?;??1??Oz????:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat|?y??0??!??f????)?1?#٠?1???NJ??:Preprocessing2U
Iterator::Model::ParallelMapV2?a?????!0|cgQ??)?a?????10|cgQ??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?U?&?5??!?r????)???c"??1??8?s???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range+*?Z^??!ͫ<c???)+*?Z^??1ͫ<c???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?u?T??!uA??J???)eS??.??1?/Z?????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch??<I?f??!jPk???)??<I?f??1jPk???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?1??8??!7?}wK0@)?J???>|?1?^??????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice???}??v?!9S?05'??)???}??v?19S?05'??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor˟ov?!???8????)˟ov?1???8????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate6?;Nё|?!?>u	?Q??)?|A	X?1???b????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?J?4Q?!Zx?G????)?J?4Q?1Zx?G????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9B&??!~??I??T??/S@Qf?>?3(6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>?$@MM@>?$@MM@!>?$@MM@      ??!       "	??9z| @??9z| @!??9z| @*      ??!       2	?-=??ɜ??-=??ɜ?!?-=??ɜ?:	?(5J@?(5J@!?(5J@B      ??!       J	g~5??g~5??!g~5??R      ??!       Z	g~5??g~5??!g~5??b      ??!       JGPUYB&??!~??b q??T??/S@yf?>?3(6@?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsE선???!E선???"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?KaX????!?s|???0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?	'z???!<?ǥ???0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradA??!???!?a?l??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterDU?W????!]?Ff??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput'?_F?=??!st?[;N??0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??w??!ر?/]??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:i???!z?O?G??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??p"Y	??!???;??"1
model/Conv1D_4/conv1dConv2DBf}?Fٚ?!|X?Nе??Q      Y@Yj+????2@a&???VRT@q<HY?$@yte??????"?

both?Your program is POTENTIALLY input-bound because 27.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?49.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 