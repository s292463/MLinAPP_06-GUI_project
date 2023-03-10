?	\:?<?@\:?<?@!\:?<?@	y??$e?@y??$e?@!y??$e?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL\:?<?@?#+????1??;?B?@A?&P?"??I???o@Y???)???rEagerKernelExecute 0*	V-?ua@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Y?????!???/>@)?1??8??1׺??z9@:Preprocessing2F
Iterator::Model ??P?\??!????s?C@)????Ϟ?1SV^ۊ5@:Preprocessing2U
Iterator::Model::ParallelMapV2q???????!ܶ>?2@)q???????1ܶ>?2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*???!????}+@)a2U0*???1????}+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateҧU??f??!?z?t-A5@)?? ??z??1?زG?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?7?Q????!iy1	?+N@)????m???1.?&Ĉ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorb?o?z?!9d]f??@)b?o?z?19d]f??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?,D?????!?!??C+7@)j?????e?1 oj,e???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9y??$e?@I?6O<?}M@QۓM?A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#+?????#+????!?#+????      ??!       "	??;?B?@??;?B?@!??;?B?@*      ??!       2	?&P?"???&P?"??!?&P?"??:	???o@???o@!???o@B      ??!       J	???)??????)???!???)???R      ??!       Z	???)??????)???!???)???b      ??!       JGPUYy??$e?@b q?6O<?}M@yۓM?A@?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad62]絮?!62]絮?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?<cZ?t??!?7??q???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradjV??N??!`??3m??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterN???Ǣ??!??>)G??0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??ņ???!]?? ???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??|?????!?Gdh???"3
model/Conv1D_1/BiasAddBiasAdd%"F?a%??!cmP????"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2ǌ?L??!I?C????"-
model/Conv1D_1/ReluRelu??mE?(??!??o?????"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?z????!?c??????Q      Y@Y?ܺ?+@a?p?h?U@q?L????@y?{?????"?
both?Your program is MODERATELY input-bound because 6.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t18.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 