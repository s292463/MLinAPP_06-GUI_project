? 	?'???*@?'???*@!?'???*@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?'???*@??@J?j@1n?R]??@A,??ypw??I?r?}ǰ@rEagerKernelExecute 0*	a;?OMm?@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapkׄ??`@!⧽?|?P@)HP?s?@1?$֋.P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??&?????!?];???>@)?ɩ?a*??1???M|=@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatT?qs*??!??`?6|??)O??D?Ư?1s?)?~J??:Preprocessing2F
Iterator::Model
+TT???!???'O @)%?S;?Ԧ?1???t????:Preprocessing2U
Iterator::Model::ParallelMapV2?????K??!???????)?????K??1???????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ?yrM???!?A?Y-{??)?ne??2??1;s???[??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat㈵? ??!?SqQ??)Œr?9>??1T??؉??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch`??5!???!?@eq?\??)`??5!???1?@eq?\??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??n?F??!`J?*d?@)u?~?1dPz?Ɖ??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???߃w?!Esy?2??)???߃w?1Esy?2??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?`"?u?!??ǜ????)?`"?u?1??ǜ????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeƢ??dpt?!???Í??)Ƣ??dpt?1???Í??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate:@0G??{?!K??JD???)??@?mX?1?}>?????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?B???T?!????l???)?B???T?1????l???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?37.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?<??kS@Q???[Q6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??@J?j@??@J?j@!??@J?j@      ??!       "	n?R]??@n?R]??@!n?R]??@*      ??!       2	,??ypw??,??ypw??!,??ypw??:	?r?}ǰ@?r?}ǰ@!?r?}ǰ@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?<??kS@y???[Q6@?"1
model/Conv1D_2/conv1dConv2D?Zm ??!?Zm ??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<?1;?I??!?tϨ????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?Hp??z??!??IM??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputc?z?c??!??N4????0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradd??7v7??!??L?????"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose??tJ????!xD??h???"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?iq.???!?qɆ?O??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?F???a??!??:?????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???=ꬡ?!???????0"3
model/Conv1D_1/BiasAddBiasAdd.??Iu??!u?G!???Q      Y@YQ^Cye4@al(????S@q????BY@yq??y???"?

both?Your program is POTENTIALLY input-bound because 40.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?37.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 