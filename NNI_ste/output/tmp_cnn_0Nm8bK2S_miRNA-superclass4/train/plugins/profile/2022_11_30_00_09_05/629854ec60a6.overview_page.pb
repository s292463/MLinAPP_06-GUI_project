?	???@?#@???@?#@!???@?#@	?eC?@?eC?@!?eC?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???@?#@?9?!??1???,?@I???8?
@Y?e??@???rEagerKernelExecute 0*     ,g@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4??@????!9}[
?@@)?????f??1?b??u?=@:Preprocessing2F
Iterator::Model?}?ƃ-??!??q?&C@)a??+e??1F?ݮ?:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?!S>??!d?F?$6@)`=?[???1??j?_P(@:Preprocessing2U
Iterator::Model::ParallelMapV24?s????!??%?'@)4?s????1??%?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??UJ????!????#@)??UJ????1????#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipN|??8G??!Dg???N@)?2p@KW??1??mӌ7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?4?ׂ?{?!??<??\@)?4?ׂ?{?1??<??\@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k?6???!?_2?7@)?[X7?i?17???v??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?22.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?eC?@I?7???u@@Q?H?)?<P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9?!???9?!??!?9?!??      ??!       "	???,?@???,?@!???,?@*      ??!       2      ??!       :	???8?
@???8?
@!???8?
@B      ??!       J	?e??@????e??@???!?e??@???R      ??!       Z	?e??@????e??@???!?e??@???b      ??!       JGPUY?eC?@b q?7???u@@y?H?)?<P@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<e?????!<e?????0"1
model/Conv1D_2/conv1dConv2D?/5GgF??!???\&&??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??=?ʧ??!w?????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?*|u??!l?,=ۨ??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?a??-E??!?I??????0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?D(????!????w???"3
model/Conv1D_1/BiasAddBiasAddM????4??!?q? ??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose˕|?>??!??@vC??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?.??????!???dtA??0"-
model/Conv1D_1/ReluRelu????M???!??Z?????Q      Y@Y??u@7?)@a%D?9?U@q!????>@y????????"?
both?Your program is POTENTIALLY input-bound because 10.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?22.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 