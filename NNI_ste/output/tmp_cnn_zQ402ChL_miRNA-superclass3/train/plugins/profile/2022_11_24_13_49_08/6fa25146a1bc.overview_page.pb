?	?Hh˹?"@?Hh˹?"@!?Hh˹?"@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Hh˹?"@0e????1?E_A?Q@A?=?
Y??I?%T`@rEagerKernelExecute 0*	??~j? r@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateq???????!?["VP@)?Nt??1???'?O@:Preprocessing2F
Iterator::Model??`ũֲ?!^y??_9@)?Pj/????1uY?$K?0@:Preprocessing2U
Iterator::Model::ParallelMapV2V}??b??!??<?p+!@)V}??b??1??<?p+!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^f?(?7??!cb???@)X9???1/?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??zO????!??Z??R@)????i??1??:?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?^??x?z?!f?fO5?@)?^??x?z?1f?fO5?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????UG??!???OYP@)ˆ5?Eag?1?"ҁ?|??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?d??~?]?!9dG{M???)?d??~?]?19dG{M???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensora??q6]?!=?;iޫ??)a??q6]?1=?;iޫ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIl?1/S@QO??;C7@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0e????0e????!0e????      ??!       "	?E_A?Q@?E_A?Q@!?E_A?Q@*      ??!       2	?=?
Y???=?
Y??!?=?
Y??:	?%T`@?%T`@!?%T`@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb ql?1/S@yO??;C7@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputW8F?Yf??!W8F?Yf??0"1
model/Conv1D_2/conv1dConv2D?9??C??!"9??????"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:?'????!;?m?c??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter7???%է?!?	
k??0"1
model/Conv1D_3/conv1dConv2D?S??x=??!6T??????"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput< ?-?l??!:?\??G??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??x7
i??!e=?P??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?CK?ܙ?!&x??????"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad?m:$?i??!?'{2??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad;??????!٫?????Q      Y@Yw?"???)@aq?{??U@q????2E@yy??2???"?
both?Your program is POTENTIALLY input-bound because 18.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?57.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?42.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 