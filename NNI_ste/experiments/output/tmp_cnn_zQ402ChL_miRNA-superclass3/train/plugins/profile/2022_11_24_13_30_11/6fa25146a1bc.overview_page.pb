?	O????@O????@!O????@	?װ?r@?װ?r@!?װ?r@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLO????@???sE)??1?O??e??A??a???I?"r?@Y?9#J{???rEagerKernelExecute 0*	??Mb?e@2F
Iterator::Model հ????!?ZNrG@)???8??1<G??f`@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;??u??!???W??@)*?"???1YjmO?;@:Preprocessing2U
Iterator::Model::ParallelMapV2j?L?:??!c?G,@)j?L?:??1c?G,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd??????!*?????J@)\?J???1??\%~"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?!S>??!????@)?!S>??1????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA.q??Ȓ?!媵??%@)}v?uŌ??1D?XZ֌@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???~?!??!(@)???~?1??!(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?m???W??!??F ??'@)???h?xd?1?O?,2???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 22.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?װ?r@I1??ZS@Q???^s4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???sE)?????sE)??!???sE)??      ??!       "	?O??e???O??e??!?O??e??*      ??!       2	??a?????a???!??a???:	?"r?@?"r?@!?"r?@B      ??!       J	?9#J{????9#J{???!?9#J{???R      ??!       Z	?9#J{????9#J{???!?9#J{???b      ??!       JGPUY?װ?r@b q1??ZS@y???^s4@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?b?????!?b?????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter`t?&ٴ?!?k??o??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??n?gc??!:???g???0"1
model/Conv1D_2/conv1dConv2Di??????!'k??d??"1
model/Conv1D_3/conv1dConv2D`?z!????!???6???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput1Qވ????!??9Fu???0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?`]b???!ܘ??A??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad&??f&O??!^?S?3??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradV5-????!?G%{L???"1
model/Conv1D_1/conv1dConv2D?!?^+P??!?<??'9??Q      Y@Y&W?+?)@a?????U@q?ʳ?{?@@y??rC???"?
both?Your program is POTENTIALLY input-bound because 22.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?53.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?33.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 