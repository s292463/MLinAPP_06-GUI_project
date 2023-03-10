?	W|C??)6@W|C??)6@!W|C??)6@	??å?????å???!??å???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCW|C??)6@? ??b??1qu ?]?3@I;?i?????Y?n?;2V??rEagerKernelExecute 0*	????xe@2F
Iterator::Model?S?[ƶ?!fuI??hJ@)???Hi??1?k?d,?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatn?B</??!O	?Ҷ?:@)ܷZ'.ǣ?1Iy#??6@:Preprocessing2U
Iterator::Model::ParallelMapV2?A?<?E??!?J?l?1@)?A?<?E??1?J?l?1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?e?I)??!????,@)?qn?1oJX??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??\7????!??Т^@)??\7????1??Т^@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip&VF#?W??!???'?G@)~b??U}?1=>	S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?t?? ?{?!2???՘@)?t?? ?{?12???՘@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap܄{eު??!V???
0@)()? ?l?1uRK֕C @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??å???I?|?0_?#@Q??܂?-V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? ??b??? ??b??!? ??b??      ??!       "	qu ?]?3@qu ?]?3@!qu ?]?3@*      ??!       2      ??!       :	;?i?????;?i?????!;?i?????B      ??!       J	?n?;2V???n?;2V??!?n?;2V??R      ??!       Z	?n?;2V???n?;2V??!?n?;2V??b      ??!       JGPUY??å???b q?|?0_?#@y??܂?-V@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFiltera?????!a?????0"1
model/Conv1D_2/conv1dConv2D?,-%???!?M9?? ??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??????!?	Z????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterc"O$m???!@?md???0"1
model/Conv1D_3/conv1dConv2D????5??!bEF?Le??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??O3?8??!}??dc???0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?dJ?? ??!?,?:x??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradV?m????!??S????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad????, ??!r*?d???"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeQ?
??q??!%??[?]??Q      Y@YD+l$Z)@a?z2~??U@q?????6@yM,!I41??"?
both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 