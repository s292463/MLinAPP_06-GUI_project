?!	?????$@?????$@!?????$@	?????@?????@!?????@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?????$@HN&n???1?H?")@AT?n.????I?Ln?@Y??????rEagerKernelExecute 0*	?p=
? ?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??};???!c????L@):?S??b??1??T&ӵK@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???N$??!|???OB@)?S?D???11?@
[A@:Preprocessing2F
Iterator::ModelT?YO???!?ל?S
@)s+??X¢?1-?P(:?@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?\?C????!?D\h*@){i? ?w??1?0???@:Preprocessing2U
Iterator::Model::ParallelMapV2????՗?!p???2??)????՗?1p???2??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??^?S??!????~???)??'c|???1?.+@9???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?p?{????!?????)???1???1P?l??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?>?6???!\-f????)?>?6???1\-f????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???1g??!??:Y?M@)??_ ??1WB???8??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??	???{?!?Bm?($??)??	???{?1?Bm?($??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????x?!??!??)????x?1??!??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????s?!??0?S???)?????s?1??0?S???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?HLP÷??!uět???)??;??~V?1h)??¶?:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?@??_?K?!#?7?T??)?@??_?K?1#?7?T??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?24.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?????@I?1??E@Q???n?I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	HN&n???HN&n???!HN&n???      ??!       "	?H?")@?H?")@!?H?")@*      ??!       2	T?n.????T?n.????!T?n.????:	?Ln?@?Ln?@!?Ln?@B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JGPUY?????@b q?1??E@y???n?I@?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter5>?E(???!5>?E(???0"1
model/Conv1D_3/conv1dConv2Dԑ)????!莒???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??w۸???!??l?qO??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?o??(???!?p%?;J??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?bD%Ϡ?!?????0"1
model/Conv1D_2/conv1dConv2D?O??!???գ???"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad??*t?֚?!???L^??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad@?7`l???!??????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?P禷}??!ƋX??_??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??p>??!???u???Q      Y@Y?$I?$I2@aܶm۶mT@q*x	r?d%@yNS!????"?
both?Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?24.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t18.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 