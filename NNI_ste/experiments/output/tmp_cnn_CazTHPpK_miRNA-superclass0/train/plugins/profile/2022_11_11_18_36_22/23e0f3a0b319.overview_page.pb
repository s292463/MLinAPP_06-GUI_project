?	\??.??@\??.??@!\??.??@	gd??@gd??@!gd??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL\??.??@?=??I???1?\7??v@A?*???ڗ?I?߃?.@Y??#??rEagerKernelExecute 0*	????xy@2F
Iterator::ModelPqx????!.s14VT@)a??+e??1xq?A?R@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ??????!7<?Xb%@)??×???1`}?d"@:Preprocessing2U
Iterator::Model::ParallelMapV2?ފ?5??!????(@)?ފ?5??1????(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Y?e0??!~5?V?@)?H?}??1??A?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceعi3NC??!?oO???@)عi3NC??1?oO???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???ɍ"??!?G3:/?2@)J?U???1"????J??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???~1{?!??<?+???)???~1{?1??<?+???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?4?;???!(?/|\@)???խ?c?1jh??. ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9fd??@I'y?i`Q@Q??E98?;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?=??I????=??I???!?=??I???      ??!       "	?\7??v@?\7??v@!?\7??v@*      ??!       2	?*???ڗ??*???ڗ?!?*???ڗ?:	?߃?.@?߃?.@!?߃?.@B      ??!       J	??#????#??!??#??R      ??!       Z	??#????#??!??#??b      ??!       JGPUYfd??@b q'y?i`Q@y??E98?;@?"1
model/Conv1D_2/conv1dConv2D???E???!???E???"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter
#y?i???!?|??ך??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?h??\??!?}S????0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?LG?׍??!T?2M????0"1
model/Conv1D_3/conv1dConv2Dϒ??\/??!??h몰??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputD?Ò????!???4????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad@?]嫝?!??j??z??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad????u9??!???J.??"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsE?????!C?gd{???"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradS\?վO??!?QwL??Q      Y@Y|??'@a|??V@q
Լ?2@@y."u83,??"?
both?Your program is POTENTIALLY input-bound because 23.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?32.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 