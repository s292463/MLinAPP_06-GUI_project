? 	u;?ʃ?!@u;?ʃ?!@!u;?ʃ?!@	Ҥ?s(?@Ҥ?s(?@!Ҥ?s(?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLu;?ʃ?!@?/??V??1??o?N???A??=^H???I? Q0c:@YΧ?UJ??rEagerKernelExecute 0*	?ʡE??@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap3P?>c??!>3???I@)ZI+?????1TXCdAH@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map%?s}???!??KX?A@)???B??10??y@@:Preprocessing2U
Iterator::Model::ParallelMapV2?b?J!??!?e?s^"@)?b?J!??1?e?s^"@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???镲??!5?gTh?@)?+-#????1??$??@:Preprocessing2F
Iterator::Model??r????!?-?+J&@)??n?????1?#[
]??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicehv?[????!D?????)hv?[????1D?????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Ց#????!r!??????)Ę??Rx??1<?s????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
???ڔ?!SJ? ??)?RB??^??1|͹????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchĕ?wF[??!I]?Q???)ĕ?wF[??1I]?Q???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??[?nK??!?Й"?J@)?,??V??1??CR????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?5???v?!^??,i???)?5???v?1^??,i???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?&?5?p?!a?~???)?&?5?p?1a?~???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?:U?g$??!*?S?a??)?E|'f?X?1_Uq???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor??ӹ??P?!\cö?̰?)??ӹ??P?1\cö?̰?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?59.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Ҥ?s(?@Iq?VUT@Q??'?Bw0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?/??V???/??V??!?/??V??      ??!       "	??o?N?????o?N???!??o?N???*      ??!       2	??=^H?????=^H???!??=^H???:	? Q0c:@? Q0c:@!? Q0c:@B      ??!       J	Χ?UJ??Χ?UJ??!Χ?UJ??R      ??!       Z	Χ?UJ??Χ?UJ??!Χ?UJ??b      ??!       JGPUYҤ?s(?@b qq?VUT@y??'?Bw0@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterҮ?b??!Ү?b??0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsZ,/??@??!????ѳ?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?Z[a?r??!??T5 ??0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??󡲊??!)??n????0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput#??9??!MB)?????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradzcj??4??!???~Rm??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradA??U?ț?!D???p???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???)-???!a.s'[.??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterL?g???!"cb?????0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?[?$?W??!?(?klr??0Q      Y@Yj+????2@a&???VRT@q-?&??D@yo????I??"?

both?Your program is POTENTIALLY input-bound because 21.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?59.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 