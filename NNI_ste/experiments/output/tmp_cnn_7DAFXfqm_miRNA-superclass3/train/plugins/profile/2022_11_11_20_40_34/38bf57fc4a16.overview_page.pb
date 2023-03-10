?	????@????@!????@	?<c?.@?<c?.@!?<c?.@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????@B_z?s??1??x>?@Aus??=A??I?????Q??Ye?`TR'??rEagerKernelExecute 0*	㥛? ?q@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??捓??!?<0??L@)??捓??1?<0??L@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?a1?Z{??!?%??a0@)*q㊋??1?Ok0D+@:Preprocessing2F
Iterator::Models??????!x?. P]1@)R臭????1yd?C??#@:Preprocessing2U
Iterator::Model::ParallelMapV2??̒ 5??!ꄛ???@)??̒ 5??1ꄛ???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???>9
??!SqJ.?N@) ???j??1)?*uU@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?I`s???!"[????T@)?V|C᳅?1\;?$?F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???8?~?!k??)??@)???8?~?1k??)??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,???c??!??-?c<O@)?O?mpf?1d,#oMM??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?22.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?<c?.@IN`E??E@Q??n?,8J@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B_z?s??B_z?s??!B_z?s??      ??!       "	??x>?@??x>?@!??x>?@*      ??!       2	us??=A??us??=A??!us??=A??:	?????Q???????Q??!?????Q??B      ??!       J	e?`TR'??e?`TR'??!e?`TR'??R      ??!       Z	e?`TR'??e?`TR'??!e?`TR'??b      ??!       JGPUY?<c?.@b qN`E??E@y??n?,8J@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???_db??!???_db??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?UWI???!?y[?V ??0"1
model/Conv1D_2/conv1dConv2D?"?)????!?E-?????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad???pg??!?????,??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradT<?-S???!~	????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose+a?Z1??!???Q??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??0̉)??!?+??ܖ??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose2?kMѠ??!I?@????"3
model/Conv1D_1/BiasAddBiasAddVlC?֠?!4"??????"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose
??\??!?8????Q      Y@YsO#,?4*@a?{a?U@q1ބO??6@y4? ?6??"?
both?Your program is POTENTIALLY input-bound because 20.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?22.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 