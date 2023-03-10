?	????|$@????|$@!????|$@	<"#?Q:@<"#?Q:@!<"#?Q:@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????|$@cG?P???1>???4?@Ac	kc??g?Ib??!?E??Y??<,Ԛ??rEagerKernelExecute 0*	Q??n7d@2F
Iterator::Model?W????!%M?޲|I@)?M)??Э?1??Z? B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2t????!V?:???;@)?#EdXţ?1|dњ??7@:Preprocessing2U
Iterator::Model::ParallelMapV2?º??Ș?!??=??-@)?º??Ș?1??=??-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??=x?Җ?!A3?㓐+@)mFA????1'\?{?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?5?ꬆ?!d?"??b@)?5?ꬆ?1d?"??b@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\8?L??!ڲ_!M?H@)0???"??15??A?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorTpxADjz?!??J??@)TpxADjz?1??J??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?,??????!??S_/@)?lɪ7i?1?=~?s??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9<"#?Q:@I??p/??5@Q???X5}R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	cG?P???cG?P???!cG?P???      ??!       "	>???4?@>???4?@!>???4?@*      ??!       2	c	kc??g?c	kc??g?!c	kc??g?:	b??!?E??b??!?E??!b??!?E??B      ??!       J	??<,Ԛ????<,Ԛ??!??<,Ԛ??R      ??!       Z	??<,Ԛ????<,Ԛ??!??<,Ԛ??b      ??!       JGPUY<"#?Q:@b q??p/??5@y???X5}R@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???d??!???d??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput???ZQ??!B૷????0"1
model/Conv1D_2/conv1dConv2DɎ?7????!?C??]p??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?Ii?:G??!\??,??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterb:??x{??!z?R???0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??Q]?N??!?O@?ZZ??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???؛V??!P?^9.???0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeH??}+??!??/ܝ???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	TransposesN??\&??!ggui???"3
model/Conv1D_1/BiasAddBiasAdd?/?тQ??!]ɢϙ???Q      Y@Y?=????(@aFX?i??U@q?=???0@y??٠?"?
both?Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?17.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 