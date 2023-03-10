?	?L?@?L?@!?L?@	?'@??@?'@??@!?'@??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?L?@a?
?+C??1?yUg????A?J!?K??Iٲ|]?@Y?*?3???rEagerKernelExecute 0*	(1?Re@2F
Iterator::Model%????g??!bi?k?J@)Pmp"????1?T??GD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat'?;???!?|?? 8@)???1???1??E%?4@:Preprocessing2U
Iterator::Model::ParallelMapV2T? ?!ǖ?!x%?r#*@)T? ?!ǖ?1x%?r#*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???tB??!?? ??2G@)???$????1:gd?~,#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??O?s'??!??)???@)??O?s'??1??)???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??-u?ד?!??j凸&@)y?Z?10??9|?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??
??x?!o?N??'@)??
??x?1o?N??'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???2W??!?N~??)@)?x?c?1?0#?t???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?54.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?'@??@I??=??R@Q??6??5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	a?
?+C??a?
?+C??!a?
?+C??      ??!       "	?yUg?????yUg????!?yUg????*      ??!       2	?J!?K???J!?K??!?J!?K??:	ٲ|]?@ٲ|]?@!ٲ|]?@B      ??!       J	?*?3????*?3???!?*?3???R      ??!       Z	?*?3????*?3???!?*?3???b      ??!       JGPUY?'@??@b q??=??R@y??6??5@?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterc??@????!c??@????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterGυz????!???q???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter+ר?!?V??R???0"1
model/Conv1D_3/conv1dConv2D?????\??!=;??M??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput*?>???!"?b?
???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputv?Yg֟??!Q+N???0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??=)U??!??u??^??"1
model/Conv1D_2/conv1dConv2D??yRU$??!Ŋ5???"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad? U3???!	6?~[???"{
\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposel?&k????!$kVz-???Q      Y@Y??u@7?)@a%D?9?U@qlW?^??A@y ??]?|??"?
both?Your program is POTENTIALLY input-bound because 20.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?54.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?35.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 