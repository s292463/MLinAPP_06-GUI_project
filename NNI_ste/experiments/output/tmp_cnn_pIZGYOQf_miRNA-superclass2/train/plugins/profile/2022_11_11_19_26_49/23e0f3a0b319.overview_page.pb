?	XSY? ~@XSY? ~@!XSY? ~@	멒\???멒\???!멒\???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLXSY? ~@SAEկt??1?s???x@AiSu?l.??I??Q???W@Y?>???ʻ?rEagerKernelExecute 0*	??ʡ)`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate???????!?ӨHQ]A@)??5????1@??c??@:Preprocessing2F
Iterator::Modelx??qo??!}E???F@)??~????1?
IT!?<@:Preprocessing2U
Iterator::Model::ParallelMapV2q??Ŗ?!S$A31@)q??Ŗ?1S$A31@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?֤????!??Z?B)@)B\9{g???1|??ɋd @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL?e?%???!???LiK@)ڐfx?18w" ?7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? x|{w?!?઩@?@)? x|{w?1?઩@?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap
?(z?c??!????kB@)d??Tkaf?1#[p?7? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?F>?x?a?!U;m????)?F>?x?a?1U;m????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor?Ws?`?^?!??̂???)?Ws?`?^?1??̂???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?19.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9멒\???IT宠k4@Q?}n4?S@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	SAEկt??SAEկt??!SAEկt??      ??!       "	?s???x@?s???x@!?s???x@*      ??!       2	iSu?l.??iSu?l.??!iSu?l.??:	??Q???W@??Q???W@!??Q???W@B      ??!       J	?>???ʻ??>???ʻ?!?>???ʻ?R      ??!       Z	?>???ʻ??>???ʻ?!?>???ʻ?b      ??!       JGPUY멒\???b qT宠k4@y?}n4?S@?"1
model/Conv1D_2/conv1dConv2D???W۾??!???W۾??"1
model/Conv1D_3/conv1dConv2D-?\?|??!???X????"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter,?#???!???m???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput#]??>]??!?Iz?W^??0"1
model/Conv1D_1/conv1dConv2D*?L????!,?D?'???"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterh???~ֈ?!?)_??Y??0"1
model/Conv1D_4/conv1dConv2DK/Q?ц?!?n?&ƴ??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??z{?B??!?Y?P???0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?5???E??!???w?R??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput#ɱ????!?i??6???0Q      Y@YZ?q?@a??ؿ??W@qD?$?F@y???-??M?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?19.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?44.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 