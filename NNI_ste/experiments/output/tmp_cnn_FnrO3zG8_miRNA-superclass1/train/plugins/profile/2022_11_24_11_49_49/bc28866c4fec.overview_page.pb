? 	???? @???? @!???? @	L6pA??@L6pA??@!L6pA??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???? @毐?r??1Hlw??@Afh<?y??I???x. @Y0??9\??rEagerKernelExecute 0*	?????c?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?_[??g??!?%????A@)B?v????11?????>@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map3?f????!|????#A@)??~1[??1????0=@:Preprocessing2U
Iterator::Model::ParallelMapV2??<+i???!?ޘy?3@)??<+i???1?ޘy?3@:Preprocessing2F
Iterator::Model?w.???!>?ƃ??9@)?-</??1ζ(`@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???,??!?;??[@)??R??F??1	0?P?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?*??p???!f"vH?D@)a?X5??1???N?L@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate%"???1??!?Y???t@)??????1<e?Q@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]R??ߔ?!n/?J?@)?I??ǌ?1?-y(????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?z?"0և?!???????)?z?"0և?1???????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?C p?y?!?a??\???)?C p?y?1?a??\???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?n???q?!??i???)?n???q?1??i???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangei??>?Qn?!?? x='??)i??>?Qn?1?? x='??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate8?q???{?!?qۥږ??)??>d?1??-?v??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?3??k?R?!s?1.92??)?3??k?R?1s?1.92??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9L6pA??@IO?????B@Q?E5?`jM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	毐?r??毐?r??!毐?r??      ??!       "	Hlw??@Hlw??@!Hlw??@*      ??!       2	fh<?y??fh<?y??!fh<?y??:	???x. @???x. @!???x. @B      ??!       J	0??9\??0??9\??!0??9\??R      ??!       Z	0??9\??0??9\??!0??9\??b      ??!       JGPUYL6pA??@b qO?????B@y?E5?`jM@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterg??H?4??!g??H?4??0"1
model/Conv1D_2/conv1dConv2DW??oe???!I?P w???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputD??A#D??!??????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad? ?H_???!,?ӹ????"1
model/Conv1D_1/conv1dConv2D??ZD5???!\_b????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????q??!vp??Я??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterW:?/?[??!??f????0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad????3ϙ?!??RQA??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad????k??!|??????"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputg\?8???!?^??~}??0Q      Y@Y??8+?!4@a??15??S@qf0?L\?0@y۬??6)??"?
both?Your program is POTENTIALLY input-bound because 12.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?23.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 