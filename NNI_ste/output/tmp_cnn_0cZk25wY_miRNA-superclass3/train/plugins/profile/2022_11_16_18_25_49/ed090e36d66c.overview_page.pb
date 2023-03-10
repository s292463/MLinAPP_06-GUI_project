?	>?
Y?@>?
Y?@!>?
Y?@	Q??8?H@Q??8?H@!Q??8?H@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL>?
Y?@ͮ{+??1?!H?@A6Y??ѝ?I?C?|@Y??=Զ??rEagerKernelExecute 0*	>
ףpyd@2F
Iterator::Model?V?Sb??!^????E@)???LL??1"???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?#F?-t??!???y??9@)???{???1?B????4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;??u???!??ApuL@)ץF?g???1??Rc?!*@:Preprocessing2U
Iterator::Model::ParallelMapV2???pY??!45?;u)@)???pY??145?;u)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?yq???!?n???~%@)?yq???1?n???~%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???$???!?M?]?/@)?'?X??17??Q4?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?}͑??!~2????@)?}͑??1~2????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??͋_??!л5??1@)@Û5x_e?1?NA?J|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?26.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Q??8?H@I????D?E@Qn?[ͧJ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ͮ{+??ͮ{+??!ͮ{+??      ??!       "	?!H?@?!H?@!?!H?@*      ??!       2	6Y??ѝ?6Y??ѝ?!6Y??ѝ?:	?C?|@?C?|@!?C?|@B      ??!       J	??=Զ????=Զ??!??=Զ??R      ??!       Z	??=Զ????=Զ??!??=Զ??b      ??!       JGPUYQ??8?H@b q????D?E@yn?[ͧJ@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?O{??(??!?O{??(??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad*t??w???!?D?[ S??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad5??
??!?Õ??"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeN?P?̣?!??"????"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose???:͛??!?l??7??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?p?ԯ??!??"???"3
model/Conv1D_1/BiasAddBiasAdd$?>?.p??!V???(???"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose???Q9???!ވ"??%??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFiltere6?-????!???z ^??0"-
model/Conv1D_1/ReluReluA?4?T1??!?K?+???Q      Y@Ym۶m۶)@a?$I?$?U@q' ???7@yg?̇????"?
both?Your program is POTENTIALLY input-bound because 16.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?26.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?23.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 