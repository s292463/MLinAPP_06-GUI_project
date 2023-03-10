?	?+,??@?+,??@!?+,??@	???Q@???Q@!???Q@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?+,??@?Li?-A??1??b?=??A?M???P??I,?9$??@Y?v?k??rEagerKernelExecute 0*	??v??ba@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatԚ?????!	|?[}A@)t`9B???1@UB??i=@:Preprocessing2F
Iterator::Modelr???7???!`??V?E@)ũ??,???1]?????;@:Preprocessing2U
Iterator::Model::ParallelMapV2Z?????!ǎS??,@)Z?????1ǎS??,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*?t??!?7?"@)*?t??1?7?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateۤ???w??!8??w?/@)?F>?x???1?d`+?(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?i???!J3?.\C@)?i???1J3?.\C@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<?D???!? ?B?L@)z?W?|?1OxKO?5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap)????u??!9u?b?1@)pz???g?1?脘5? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?54.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???Q@I(I?B?S@Q?xT?|1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Li?-A???Li?-A??!?Li?-A??      ??!       "	??b?=????b?=??!??b?=??*      ??!       2	?M???P???M???P??!?M???P??:	,?9$??@,?9$??@!,?9$??@B      ??!       J	?v?k???v?k??!?v?k??R      ??!       Z	?v?k???v?k??!?v?k??b      ??!       JGPUY???Q@b q(I?B?S@y?xT?|1@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???Ѿ???!???Ѿ???0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsPk??Ȳ??!?/???6??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?3	r}??!?d`?????0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?x??0ƥ?!? 5,??"1
model/Conv1D_3/conv1dConv2D?? ?Ii??!???q????"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput????R??!Q?<?M??0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?	?k??!Ê?????0"1
model/Conv1D_2/conv1dConv2D\D?t;??!?
LX}???"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?h??I???!x??????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradi?'x??!?-??U??Q      Y@Y     ?-@a     JU@qTO??A@yL??????"?
both?Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?54.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?35.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 