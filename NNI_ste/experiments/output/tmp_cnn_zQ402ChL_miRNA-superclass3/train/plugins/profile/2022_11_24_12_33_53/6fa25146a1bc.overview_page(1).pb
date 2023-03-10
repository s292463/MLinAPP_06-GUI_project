?	??????d@??????d@!??????d@	Hؕ?Z???Hؕ?Z???!Hؕ?Z???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??????d@?|A	?X@1U?]??H@A40??&??I()? ?.@Y?3M?~2??rEagerKernelExecute 0*	?Zd?d@2F
Iterator::Model??r????!???MUG@)???|\??1?S{R?>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate4?/.Ui??!???q?@@)?J[\?3??1???Ԓz=@:Preprocessing2U
Iterator::Model::ParallelMapV2????̓??!JD?? 0@)????̓??1JD?? 0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?a̶ۢ?!41 ???J@)?b?T4֎?1W̒??"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat#?k$	??!X_]8?%@)?hE,b??1K??F?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorl{?%9`w?!8?Fx?W@)l{?%9`w?18?Fx?W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapɓ?k&߬?!I&?}??@@)OYM?]g?1?&??S??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor??m3?a?!??q????)??m3?a?1??q????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceM??ua?!yK?tIk??)M??ua?1yK?tIk??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Gؕ?Z???Ixr??fQ@QJ,???L>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|A	?X@?|A	?X@!?|A	?X@      ??!       "	U?]??H@U?]??H@!U?]??H@*      ??!       2	40??&??40??&??!40??&??:	()? ?.@()? ?.@!()? ?.@B      ??!       J	?3M?~2???3M?~2??!?3M?~2??R      ??!       Z	?3M?~2???3M?~2??!?3M?~2??b      ??!       JGPUYGؕ?Z???b qxr??fQ@yJ,???L>@?"1
model/Conv1D_2/conv1dConv2D5?G????!5?G????"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??E?(??!???FC???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???Ǟ/??!??SU	???0"1
model/Conv1D_3/conv1dConv2D?a?˳?!e?sa?I??"1
model/Conv1D_4/conv1dConv2D???J?^??!??JXU??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??????!i??;!??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter[?)lFV??!????~???0"1
model/Conv1D_1/conv1dConv2DcMu?<ݩ?!???qR???"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??=>????!???5???0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterI?A?ۻ??!??&?????0Q      Y@Y?">?Tr@a?????W@q?a?1n?@@yAL>k>l?"?
both?Your program is POTENTIALLY input-bound because 60.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?33.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 