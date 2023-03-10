?	
If?G@
If?G@!
If?G@	?*?	??@?*?	??@!?*?	??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL
If?G@?T1????1|{נ/}@A?磌? ??IZ??ڊ?@Y??JY?8??rEagerKernelExecute 0*	??(\?Hp@2U
Iterator::Model::ParallelMapV2}A	]??!???nɈH@)}A	]??1???nɈH@:Preprocessing2F
Iterator::Modelĳ??!"!?ӍP@)????ߦ?1Y5???%1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$}ZE??!??? Q]2@)?5?eܤ?1?>?F/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???C?r??!?t??G @)???C?r??1?t??G @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?R?{/??!߄q??&@)?z??9y??1??aN?2
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;7m?i???!??KX?@@)??Cl??1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorCqǛ?}?!?Ue
??@)CqǛ?}?1?Ue
??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??4}v??!l`?ܮ(@)??}???e?1?g܆o??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?*?	??@I?A?i??L@Q?K??۱C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?T1?????T1????!?T1????      ??!       "	|{נ/}@|{נ/}@!|{נ/}@*      ??!       2	?磌? ???磌? ??!?磌? ??:	Z??ڊ?@Z??ڊ?@!Z??ڊ?@B      ??!       J	??JY?8????JY?8??!??JY?8??R      ??!       Z	??JY?8????JY?8??!??JY?8??b      ??!       JGPUY?*?	??@b q?A?i??L@y?K??۱C@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?̲V???!?̲V???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputU=?????!??/?????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter52a
l???!0?T??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad I?U????!La??O??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad=PZN???!?z?5aG??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??wIBĤ?!is?~????0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsy e?٢?!xi?!;??"1
model/Conv1D_2/conv1dConv2D9?]??.??!???@????"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?6???!??!?7)?(???0"1
model/Conv1D_3/conv1dConv2D?x܊????!?Ƅ?????Q      Y@Y?ܺ?+@a?p?h?U@q?`?? ?=@y??A?B???"?
both?Your program is POTENTIALLY input-bound because 15.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?41.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?29.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 