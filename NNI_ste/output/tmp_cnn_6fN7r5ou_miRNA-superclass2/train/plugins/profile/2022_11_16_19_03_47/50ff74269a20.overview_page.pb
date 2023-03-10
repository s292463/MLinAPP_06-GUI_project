?	?=&R?M@?=&R?M@!?=&R?M@	????M2@????M2@!????M2@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?=&R?M@u?R??1n??)o@A?n?l???Iy=?? @Yȳ˷>???rEagerKernelExecute 0*	y?&1?g@2F
Iterator::ModelgF?N???!JEM??)H@)[&??|??1?D8??h@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?|?5^???!?ݾ??W:@)????u???1g?͸?7@:Preprocessing2U
Iterator::Model::ParallelMapV20???DK??! T?/@)0???DK??1 T?/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?W????!?c㷛%@)?W????1?c㷛%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??n??;??!???:?I@)?l?%????1j????* @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?? 4J???!?7???R/@)????????1f0?.?m@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?w??Dgy?!?Q???
@)?w??Dgy?1?Q???
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?g$B#ؠ?!?J?gA?1@)??@???h?1Y?B??_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?28.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????M2@I?)-??F@QS?)?	?I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	u?R??u?R??!u?R??      ??!       "	n??)o@n??)o@!n??)o@*      ??!       2	?n?l????n?l???!?n?l???:	y=?? @y=?? @!y=?? @B      ??!       J	ȳ˷>???ȳ˷>???!ȳ˷>???R      ??!       Z	ȳ˷>???ȳ˷>???!ȳ˷>???b      ??!       JGPUY????M2@b q?)-??F@yS?)?	?I@?"1
model/Conv1D_2/conv1dConv2D??n?K??!??n?K??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:?p&???!Щ?o[???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?ى?F??!p????D??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput[?lr=??!?va? j??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?v;??!@O?dk??0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter)?q0???!???U??0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?O.?Y??!c;??0"1
model/Conv1D_3/conv1dConv2D?Q?z?[??!)?? ??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?Y9Ti???!?yP?}???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad#????!??pN?J??Q      Y@Y?=????(@aFX?i??U@qϩ???O>@y?N?Pa??"?
both?Your program is POTENTIALLY input-bound because 16.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?28.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 