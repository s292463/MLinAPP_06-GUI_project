?	Έ????@Έ????@!Έ????@	????[@????[@!????[@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLΈ????@???$????1e?<???A,cC7???I??H???@Y?z?΅??rEagerKernelExecute 0*	K7?A`?c@2F
Iterator::Model*7QKs+??!?A?0WI@)tB??K8??1Q?? ~A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatzUg????!>B??9?7@))x
?R??1:???j3@:Preprocessing2U
Iterator::Model::ParallelMapV2?Wt?5=??!??w?.@)?Wt?5=??1??w?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceV??f?\??!?D?G? (@)V??f?\??1?D?G? (@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatec?: ⮞?!7??}3@)?e3????1?f1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?e?I)??!1?TϨ?H@)}?H?F???1?h?e??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Z?!?{?!????&@)?Z?!?{?1????&@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvQ??Ǡ?!?_???4@)0?AC?g?1?[?ʉ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????[@I?P?0?O@Q]??0??@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???$???????$????!???$????      ??!       "	e?<???e?<???!e?<???*      ??!       2	,cC7???,cC7???!,cC7???:	??H???@??H???@!??H???@B      ??!       J	?z?΅???z?΅??!?z?΅??R      ??!       Z	?z?΅???z?΅??!?z?΅??b      ??!       JGPUY????[@b q?P?0?O@y]??0??@@?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter΋?????!΋?????0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad1??s?Ʀ?! ?O?6??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?????΢?!?	"??N??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?\??Kc??!? M??g??0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???????!????I<??0"1
model/Conv1D_1/conv1dConv2Dx?))?)??!S2$z???"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits7/[??!M
??????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradV	????!?kZ????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???P???!?XW?0v??"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?K?;?˚?!X??"??Q      Y@Yyxxxxx*@a??????U@q?UR??6@yC?o????"?
both?Your program is POTENTIALLY input-bound because 12.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?49.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 