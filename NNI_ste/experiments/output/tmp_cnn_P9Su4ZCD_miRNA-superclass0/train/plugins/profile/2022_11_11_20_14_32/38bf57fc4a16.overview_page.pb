? 	?tB??$@?tB??$@!?tB??$@	B(?Oܺ??B(?Oܺ??!B(?Oܺ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?tB??$@Ul???c@1.?l?I&@At	?????I???5?E@Y?~??@???rEagerKernelExecute 0*	??????@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????O??!
8,%I@)?~?????1om?o??H@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?Բ??H??!?T?9D?F@)GW?????1VQ_6??E@:Preprocessing2F
Iterator::Model?q???[??!?????@)??Ma????1қ?Li @:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??q?????!*] l????)??????1ğ?<???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd ?.????!?0?D???)??dV?p??1?,Y'????:Preprocessing2U
Iterator::Model::ParallelMapV2?q?Pi??!???*?E??)?q?Pi??1???*?E??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?F<?͌??!_?????)Z?N????12??W???:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?
?Ov??!.?O????)?
?Ov??1.?O????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???O????!9{??I@)h??n?|?1??/]P???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???<z?!?tT$??)???<z?1?tT$??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?????u?!1?????)?????u?11?????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??rf?Bo?!1??GZ??)??rf?Bo?11??GZ??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?n??S}?!??^4?'??)4Lm???^?1
?.???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?'?>?I?!??Ď???)?'?>?I?1??Ď???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?26.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9B(?Oܺ??I?S?]?G@Q?H/??\I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ul???c@Ul???c@!Ul???c@      ??!       "	.?l?I&@.?l?I&@!.?l?I&@*      ??!       2	t	?????t	?????!t	?????:	???5?E@???5?E@!???5?E@B      ??!       J	?~??@????~??@???!?~??@???R      ??!       Z	?~??@????~??@???!?~??@???b      ??!       JGPUYB(?Oܺ??b q?S?]?G@y?H/??\I@?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter%wv?<???!%wv?<???0"1
model/Conv1D_3/conv1dConv2D?ܯ??q??!*???+??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?D-O?խ?!<{^?????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterQ???????!P???d???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput???R>???!y 2?????0"1
model/Conv1D_2/conv1dConv2D??????!???K???"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad???V???!^R???y??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?rhP??! ,?????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad$Qq@??!B<???r??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?lf?
??!*??????Q      Y@Y?$I?$I2@aܶm۶mT@qB????$@y?3s????"?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?26.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 