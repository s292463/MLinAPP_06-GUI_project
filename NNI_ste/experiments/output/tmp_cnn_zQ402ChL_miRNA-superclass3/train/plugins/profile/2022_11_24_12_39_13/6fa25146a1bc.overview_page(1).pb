?!	3T?T?9 @3T?T?9 @!3T?T?9 @	?S?dg!@?S?dg!@!?S?dg!@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL3T?T?9 @???_v? @1?P?l?@A,?V]?j??I?*?3?@Y??????rEagerKernelExecute 0*	?&1??@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapG?P?[??!?_?}I@)Hj?drj??1?L&?a?G@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?q?????!??a??B@)a???)??1Y???c@@:Preprocessing2F
Iterator::Modely7R?H??!Y?9E"@)?πz3j??1 ???G?@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??W????!5??*3?@)???^???1b??R??@:Preprocessing2U
Iterator::Model::ParallelMapV2?r?SrN??!cg???@)?r?SrN??1cg???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(???o??!????????)??5&Č?1??TY3b??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???죓?!,_=?<???):!t?%??1H??%p9??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?:8؛??!?????k??)?:8؛??1?????k??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipH?V???!*`!??J@)?U?????1nO?.???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???jdWz?!ɛ?֘i??)???jdWz?1ɛ?֘i??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?1=a?t?!??n?H??)?1=a?t?1??n?H??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range'0??mp?!A?U????)'0??mp?1A?U????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate??-</{?!??K???)???k?6\?1?xpi!???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor??!??J?!????F???)??!??J?1????F???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?33.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t25.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?S?dg!@I?+??M@Q??Fc*@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???_v? @???_v? @!???_v? @      ??!       "	?P?l?@?P?l?@!?P?l?@*      ??!       2	,?V]?j??,?V]?j??!,?V]?j??:	?*?3?@?*?3?@!?*?3?@B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JGPUY?S?dg!@b q?+??M@y??Fc*@@?"1
model/Conv1D_2/conv1dConv2D?EN`*??!?EN`*??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputi,?v????!
?"?O???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter[3N&?.??!\?$o[???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?????'??!h;?YI??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad"??{ۂ??!?z?U????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?i?.?d??!?G?[L???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??'B???!`G?c???"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad?l ???!/ne:???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter>aW?7??!C??????0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??)|?!??!A=?i"??Q      Y@YEN?ƾ3@ao?{DNT@qW???3%@y??aKIR??"?
both?Your program is MODERATELY input-bound because 8.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?33.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t25.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 