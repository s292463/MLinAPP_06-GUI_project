? 	?]?o? @?]?o? @!?]?o? @      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?]?o? @?uS?ke??1?^(`;8@A>]ݱ?&??I???|@@@rEagerKernelExecute 0*	??S㥎?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?<?+J	??!??9bg?E@)5ӽN????1u2_?C@:Preprocessing2F
Iterator::Model }??A???!??D?>@)?w??x[??1?Uz(?2;@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??h?????!????j5@)^gE?D??1?+?@*/@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatW?'???!R??~@)?BX?%???1???͋@:Preprocessing2U
Iterator::Model::ParallelMapV2?)??F???!?Ř??@)?)??F???1?Ř??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN??????!ЋO??@)?cϞ˔?1T??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???U???!P???G@)??PN????1~?0dӒ @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;??Tގ??!??s(v @)5&?\R??1!'???2??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?&S???!{I5???)?&S???1{I5???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ʅʿ?w?!!??ڋs??)?ʅʿ?w?1!??ڋs??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range :̗`o?!????D1??) :̗`o?1????D1??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?,{?l?!???????)?,{?l?1???????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate&U?M?Ms?!^.?1!1??)??A??S?1(0!wj??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensory?&1?L?!?o?H>???)y?&1?L?1?o?H>???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?48.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIkx???hQ@QU%?]>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?uS?ke???uS?ke??!?uS?ke??      ??!       "	?^(`;8@?^(`;8@!?^(`;8@*      ??!       2	>]ݱ?&??>]ݱ?&??!>]ݱ?&??:	???|@@@???|@@@!???|@@@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qkx???hQ@yU%?]>@?"1
model/Conv1D_2/conv1dConv2DL??w???!L??w???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputq-??A???!? ťܸ??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??:???!??!a??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?}-??	??!???[???"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGraddWV??ס?!???N???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad-?
QH???!???0??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad??19Ξ?!??A????"1
model/Conv1D_1/conv1dConv2DG??趞?!?/?0I	??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?$?????!R|;1???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterV>@?~??!7?Y)ɣ??0Q      Y@Y?-??-?3@a?4H?4T@q?Fh?17@yQ??????"?
both?Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?48.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?23.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 