?	ǝ???#@ǝ???#@!ǝ???#@	?l??????l?????!?l?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLǝ???#@?D?? ??1??p?Q?@A[@h=|???I3?`???Y?
~b???rEagerKernelExecute 0*	????Ƨh@2F
Iterator::Model???X???!?R??ZD@)???????1??#SrQ=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"m?OT6??!??=1??;@)???????1N&?48@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ÖM???!?=_???4@)'?y?3??1?E?d?,@:Preprocessing2U
Iterator::Model::ParallelMapV2?????![?v?&@)?????1[?v?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip`?eM,???!?r?)?M@)?&S???1f7?'St"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??P?n??!????].@)??P?n??1????].@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorE?u????!?k?d?c@)E?u????1?k?d?c@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?OU??X??!^V}r? 6@)??+H3f?12??A????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?l?????I??d@@Q?ǥ?VP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D?? ???D?? ??!?D?? ??      ??!       "	??p?Q?@??p?Q?@!??p?Q?@*      ??!       2	[@h=|???[@h=|???![@h=|???:	3?`???3?`???!3?`???B      ??!       J	?
~b????
~b???!?
~b???R      ??!       Z	?
~b????
~b???!?
~b???b      ??!       JGPUY?l?????b q??d@@y?ǥ?VP@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter]?.	A???!]?.	A???0"1
model/Conv1D_2/conv1dConv2D???@T???!1&?)?8??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputUP	?????!#=/(????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??p??Ŧ?!?R=E????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??????!??^??{??"1
model/Conv1D_3/conv1dConv2Dفx?Cҡ?!???q)???"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?6|?cn??!E??????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?)??#??!?ɡ?Mo??0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???f֚?!CZ??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputwP?X???!?M?????0Q      Y@YD+l$Z)@a?z2~??U@q?6?IZ?@ytm?4f??"?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?17.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 