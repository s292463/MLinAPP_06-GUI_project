?	 ???W!@ ???W!@! ???W!@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC ???W!@??? @1#/kb??AG;n??t??I?U??k@rEagerKernelExecute 0*	l????"^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????ߥ?!???z?A@)m??)嵢?1*???P>@:Preprocessing2F
Iterator::Model??b?=??!?Z??rD@)?]~p??1H?"?Ҩ8@:Preprocessing2U
Iterator::Model::ParallelMapV2?Yh?4??!??v?<0@)?Yh?4??1??v?<0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???4cф?!"M"l? @)???4cф?1"M"l? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&r?????!??=9??.@)?% ??*??1? [.>?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf???-=??!`?h?&?M@)Ψ?*??}?1 ??=.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorKr??&Oy?!T?D݀@)Kr??&Oy?1T?D݀@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????㾕?!QPB?ӝ1@)????=f?13L?p?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?58.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?L?;??T@Qw?j??0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??? @??? @!??? @      ??!       "	#/kb??#/kb??!#/kb??*      ??!       2	G;n??t??G;n??t??!G;n??t??:	?U??k@?U??k@!?U??k@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?L?;??T@yw?j??0@?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???A@m??!???A@m??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad\F? ??!gg?~????"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits??܌?
??!f?Ѳ???"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput;X????!?t??????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??A̝?!V??5B???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?$
Oi???!????????0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad|??s??!|?[c$j??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??{ ?ڜ?!??c??7??0"1
model/Conv1D_2/conv1dConv2D?I?^???!
!?q????"}
^gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?yO331??!????4??Q      Y@Y?JG?(@a??7a?U@q?v/?CK@y?/L??"?
both?Your program is POTENTIALLY input-bound because 24.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?58.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?54.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 