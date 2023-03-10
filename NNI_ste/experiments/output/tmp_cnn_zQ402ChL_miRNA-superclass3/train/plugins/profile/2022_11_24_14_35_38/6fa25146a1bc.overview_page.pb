?	C?*q?@C?*q?@!C?*q?@	?c%q??6@?c%q??6@!?c%q??6@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLC?*q?@????? @1?)?n???A??V*??I^G??t@Y?t??ϡ??rEagerKernelExecute 0*	8?A`?4b@2F
Iterator::Modelh??52??!?h?l?I@)½2o?u??1?گBbf@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat}y?ѩ??!6??^:@)_ Q??1cC)? ?5@:Preprocessing2U
Iterator::Model::ParallelMapV2??iܛ?!??\?2@)??iܛ?1??\?2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM??????!|ɇ?Ҹ@)M??????1|ɇ?Ҹ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatemr??	??!;W?^?-@)?^Pj??1????_@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipŎơ~??!???BH@)?6qr?C??1???C?&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Xl???z?!OC%??@)?Xl???z?1OC%??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapɬ??vh??!?r?k]0@)??1??b?1?n3?s??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t26.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?c%q??6@I??I\K@Qvz??Y?6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????? @????? @!????? @      ??!       "	?)?n????)?n???!?)?n???*      ??!       2	??V*????V*??!??V*??:	^G??t@^G??t@!^G??t@B      ??!       J	?t??ϡ???t??ϡ??!?t??ϡ??R      ??!       Z	?t??ϡ???t??ϡ??!?t??ϡ??b      ??!       JGPUY?c%q??6@b q??I\K@yvz??Y?6@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput\?Q{⢻?!\?Q{⢻?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?e:h?1??!|?qj???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter ??٘U??!>?T???0"1
model/Conv1D_2/conv1dConv2D?!/N??!??r???"1
model/Conv1D_3/conv1dConv2D?ҼI???! ?1?2 ??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?????!??4?(U??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?8?0????!??Rjh??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?!z??l??!0>&?u??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad0k??????!?TQN?C??"1
model/Conv1D_1/conv1dConv2DYڳ?3>??!????????Q      Y@Y&W?+?)@a?????U@q??????&@y?[\?r??"?
host?Your program is HIGHLY input-bound because 22.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t26.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 