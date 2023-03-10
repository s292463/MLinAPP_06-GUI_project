?	? U??@? U??@!? U??@	
????h@
????h@!
????h@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL? U??@
.V?`??1D6?.6m??AZ?!?[=??Ic}??@Y???%???rEagerKernelExecute 0*	?MbX_q@2U
Iterator::Model::ParallelMapV2?n-????!I?MPXM@)?n-????1I?MPXM@:Preprocessing2F
Iterator::Model?JY?8???!S{??'R@)???c?ң?1???{?+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!?x7v??/@)*? ?hU??1)Fؑ++@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceyxρ???!???@Q?@)yxρ???1???@Q?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQg?!?{??!????a;@)???Co??1??__?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??%P6??!@M?f?@)q?{??c??16!??)p@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Q,??z?!s??w??@)??Q,??z?1s??w??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?c*?ߗ?!?*v?? @)??_?Le?1ۦ}????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?51.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????h@I:?R?r?R@Q?Ƭ??4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
.V?`??
.V?`??!
.V?`??      ??!       "	D6?.6m??D6?.6m??!D6?.6m??*      ??!       2	Z?!?[=??Z?!?[=??!Z?!?[=??:	c}??@c}??@!c}??@B      ??!       J	???%??????%???!???%???R      ??!       Z	???%??????%???!???%???b      ??!       JGPUY????h@b q:?R?r?R@y?Ƭ??4@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter75DҞ??!75DҞ??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??R???!1A??^??0"1
model/Conv1D_2/conv1dConv2D䈅?2??!҂kr|??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?_?M4???!????P???0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits? g?????!??_;???"1
model/Conv1D_3/conv1dConv2Dq?֠'???!??z@???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?ԶU???!?6U?J??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?գ??L??!?s?N???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad#??ӎ??!vh????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??`"??!i?w&3??Q      Y@Y_?.=?'@a??)Z8V@qؼ++ɉ;@y????V???"?
both?Your program is POTENTIALLY input-bound because 21.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?51.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 