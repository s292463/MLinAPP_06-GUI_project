?	'l???$@'l???$@!'l???$@	$wv?8?	@$wv?8?	@!$wv?8?	@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL'l???$@Y?e0F$??1??z6+@A??9"ߥ??I??5|?@Y?3iSu???rEagerKernelExecute 0*	?MbX?f@2F
Iterator::ModelPS???"??!1?n?F@)???8+???1؁??j}@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat,???c??!+??qi<@)?#ӡ????1?????8@:Preprocessing2U
Iterator::Model::ParallelMapV27p??G??!f??))@)7p??G??1f??))@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?je?/???!?&?Y?|%@)?je?/???1?&?Y?|%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??$[]N??!????>K@)?Ց#????1?r?3??"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??G??5??!?????^.@)???s????1??S*M?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?@H0?{?!˓????@)?@H0?{?1˓????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[y??????!g_?A?0@)???e?1?
_?~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9$wv?8?	@I޸?'@Q@Q?M?M??;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Y?e0F$??Y?e0F$??!Y?e0F$??      ??!       "	??z6+@??z6+@!??z6+@*      ??!       2	??9"ߥ????9"ߥ??!??9"ߥ??:	??5|?@??5|?@!??5|?@B      ??!       J	?3iSu????3iSu???!?3iSu???R      ??!       Z	?3iSu????3iSu???!?3iSu???b      ??!       JGPUY$wv?8?	@b q޸?'@Q@y?M?M??;@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@???е?!@???е?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradp]?Z???!^w?"x???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradW?`?&M??!4???????"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?#6V???!?x_ZW??0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose8k??R???!щA?=??"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposebk7???!??.?k???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??R????!?F?+lp??0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose??Y5???!??2???"3
model/Conv1D_1/BiasAddBiasAdd?d?????!???1W??"-
model/Conv1D_1/ReluReluu?7Ֆ??!???3???Q      Y@Yyxxxxx*@a??????U@q(S?8??7@y??!??y??"?
both?Your program is POTENTIALLY input-bound because 11.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?57.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?23.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 