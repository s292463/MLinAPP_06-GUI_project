? 	?H?+`!@?H?+`!@!?H?+`!@	??HO??@??HO??@!??HO??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?H?+`!@5????k??1G?ŧ ???A??ޫVf??I?`<c@Y?Y-??D??rEagerKernelExecute 0*	|?Ga?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?'?H0???!?Q5WkVP@)?t?yƾ??1p5NiO@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?*mq????!F;]?:@)???|?r??1?WYO??6@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat????5???!??? ?@)?W?2ı??1:%??=@:Preprocessing2F
Iterator::Model?0Bx?q??!<?g?JZ@)$~?.r??1.D)?@:Preprocessing2U
Iterator::Model::ParallelMapV2????:q??!J??l4@)????:q??1J??l4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?;? ??!?5:
????)???v?
??1?i?????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat հ????!??]u???)????Đ??1 ???p??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?E&??H??!??9?*??)?E&??H??1??9?*??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd???????!+?Y??P@)??E?>??1l?3Wp???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?R]??{?!?lK??Z??)?R]??{?1?lK??Z??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Ŋz?!?f|}????)????Ŋz?1?f|}????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeZ?rL?o?!	??B???)Z?rL?o?1	??B???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate???j??!????l??)?[[%X\?1ֵH??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor???ْUQ?!??l?sθ?)???ْUQ?1??l?sθ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?58.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??HO??@IU?a?XT@Qu??p?@0@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5????k??5????k??!5????k??      ??!       "	G?ŧ ???G?ŧ ???!G?ŧ ???*      ??!       2	??ޫVf????ޫVf??!??ޫVf??:	?`<c@?`<c@!?`<c@B      ??!       J	?Y-??D???Y-??D??!?Y-??D??R      ??!       Z	?Y-??D???Y-??D??!?Y-??D??b      ??!       JGPUY??HO??@b qU?a?XT@yu??p?@0@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?B??????!?B??????0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?7}`A???!U=?????0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits5o??Ǣ?!??#??	??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?0P??!???]??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?gN6????!ݴе?N??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter"????!??!??5?2??0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??d~I??!??????0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???S&???!x?
hx???"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad?w=?ކ??!???W????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter-|??c???!?]w?,J??0Q      Y@Y????B2@a?DZ/`T@qS*?r??0@y??)ؤ???"?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?58.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 