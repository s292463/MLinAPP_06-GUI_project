?	?\S ?c@?\S ?c@!?\S ?c@	A??|F? @A??|F? @!A??|F? @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?\S ?c@???t???1Zg|_\
@AF?7?k???I??N??@Y?y?W??rEagerKernelExecute 0*	X9??bd@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?pX?Q??!?b"??A@)#1A?ª?1x?,?J@@:Preprocessing2F
Iterator::Model$}ZE??!???T+G@)1]??a??1?o?
Z2=@:Preprocessing2U
Iterator::Model::ParallelMapV2ӥI*S??!C?k???0@)ӥI*S??1C?k???0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????-??!??j|m?%@){?????1̈́?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?CV?z??!Ik???J@)C?(^em??1????D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensortCSv?A}?!??P???@)tCSv?A}?1??P???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ
,?)??!1?>T??B@)??9]k?1զ?S6 @:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor ?o_?i?!.Ű5\???) ?o_?i?1.Ű5\???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?R?G^?!R???T!??)?R?G^?1R???T!??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9A??|F? @I??=?Q?K@Q+????E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???t??????t???!???t???      ??!       "	Zg|_\
@Zg|_\
@!Zg|_\
@*      ??!       2	F?7?k???F?7?k???!F?7?k???:	??N??@??N??@!??N??@B      ??!       J	?y?W???y?W??!?y?W??R      ??!       Z	?y?W???y?W??!?y?W??b      ??!       JGPUYA??|F? @b q??=?Q?K@y+????E@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?(?0˵?!?(?0˵?0"1
model/Conv1D_2/conv1dConv2D?m??ž??!T????D??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradbSil????!, ?L?B??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad????v??!l?N?I???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??r?=L??!??U??Y??0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose???-??!??KR???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??F???!???????"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose??? 67??!θ???T??"3
model/Conv1D_1/BiasAddBiasAdd-א?v???!??z?i???"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?L??ˠ?!?l??????Q      Y@YV?H??*@a?d?v??U@q,???`?8@yM?iI+???"?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?24.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 