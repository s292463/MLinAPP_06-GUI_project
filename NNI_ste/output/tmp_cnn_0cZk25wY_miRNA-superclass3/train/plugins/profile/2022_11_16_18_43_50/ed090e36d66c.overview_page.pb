?	*?J=R@*?J=R@!*?J=R@	?R?P??@?R?P??@!?R?P??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL*?J=R@?K?KƱ??1(??Z&???A?o????I?Jvl@Y*?Z^???rEagerKernelExecute 0*	?"??~?c@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj?????!?t???B@)˟o???1z?ԯ?@@:Preprocessing2F
Iterator::Model??????!f?:??qF@)O???Ш?1???Pg?>@:Preprocessing2U
Iterator::Model::ParallelMapV2?u??!??jC??+@)?u??1??jC??+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice\?	??b??!?qv<?@)\?	??b??1?qv<?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?N\?W ??!?b?U?K@)<-?p?'??1f?C^m@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate~?Az???!???P?&@)>?????}?1??u?z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr??	?z?!??\??@)r??	?z?1??\??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(]????!??qH??)@)*T7?c?1IC?{????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?51.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?R?P??@I1?tH?R@Q?Ltnk9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?K?KƱ???K?KƱ??!?K?KƱ??      ??!       "	(??Z&???(??Z&???!(??Z&???*      ??!       2	?o?????o????!?o????:	?Jvl@?Jvl@!?Jvl@B      ??!       J	*?Z^???*?Z^???!*?Z^???R      ??!       Z	*?Z^???*?Z^???!*?Z^???b      ??!       JGPUY?R?P??@b q1?tH?R@y?Ltnk9@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?T?Bӱ?!?T?Bӱ?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?v???l??!
?&?	??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradB??j??!??W?j???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputwA?cB??!N&(?p??0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeuSZו??!+??li0??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?V(l̢?!?9?;????"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?>?ޠd??!?eWV>??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFiltermG?g*??!y?=8????0"3
model/Conv1D_1/BiasAddBiasAdd??Iv??!?Jqr???"-
model/Conv1D_1/ReluRelu<??h?t??!?=?????Q      Y@Ym۶m۶)@a?$I?$?U@q̐?TZ#C@y??2jI??"?
both?Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?51.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?38.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 