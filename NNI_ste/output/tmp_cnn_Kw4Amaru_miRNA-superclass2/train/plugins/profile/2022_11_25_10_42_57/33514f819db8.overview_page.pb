?	? [??@? [??@!? [??@	N=1?K?@N=1?K?@!N=1?K?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL? [??@?ui???1??yr?	@AV??;Mf??IϿ]??. @Y?D????rEagerKernelExecute 0*	>
ףpe@2F
Iterator::Modelh@?5_??!???|A?H@)zPP?V???1?&Q=?aA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat@KW??x??!????H?>@)ŏ1w-!??1t?g???:@:Preprocessing2U
Iterator::Model::ParallelMapV2?`TR'???!?fv???-@)?`TR'???1?fv???-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_
?]???!??XJ?@)_
?]???1??XJ?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate:τ&???!?/???)@)?o???1?ub|?Y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??mP????!O???-I@)W#?҂?1??ax?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-??z?!Dz?rv@)??-??z?1Dz?rv@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape5]Ot]??!y??,L,@)\qqTn?f?1񛼺?I??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?34.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9M=1?K?@I????D@Qq???)K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ui????ui???!?ui???      ??!       "	??yr?	@??yr?	@!??yr?	@*      ??!       2	V??;Mf??V??;Mf??!V??;Mf??:	Ͽ]??. @Ͽ]??. @!Ͽ]??. @B      ??!       J	?D?????D????!?D????R      ??!       Z	?D?????D????!?D????b      ??!       JGPUYM=1?K?@b q????D@yq???)K@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter'?;??Y??!'?;??Y??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter\????t??!U?[?M???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?%?y????!#?}?s??0"1
model/Conv1D_2/conv1dConv2D???v????!`"??8???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?1J?➡?!fWK????"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam?Wb?(a??!e???+??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputЅH?0??!? ?61??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradN?[?????!?q6????"1
model/Conv1D_3/conv1dConv2DԉZ&3???!Q??????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeJk&??Ü?!?~j????Q      Y@YAd?W?,)@ax??g?U@q?:??y@@y?b[.'??"?
both?Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?34.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?33.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 