?	Gɫs?@Gɫs?@!Gɫs?@	/?H?3&@/?H?3&@!/?H?3&@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCGɫs?@0??L?^??1K9_??x@Iù????YGY???.??rEagerKernelExecute 0*	?$???@2U
Iterator::Model::ParallelMapV2?~j?t??!As???Q@)?~j?t??1As???Q@:Preprocessing2F
Iterator::Model>??<??!s??2!?V@)vöE???1ʔY?|n3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO!W?Y??!??~?g@)x` ?C???1????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???1??!9,???1@)???1??19,???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?f׽???!p?Y??@)eȱ????1?Zf?IU @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???׼?!g<?h??"@)????[??1?%V$???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??J?8|?!??%????)??J?8|?1??%????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`?_????!Z?e-?@)ŭ???g?1R??aR??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?24.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9.?H?3&@I?i??ҭ:@Q?b??$O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0??L?^??0??L?^??!0??L?^??      ??!       "	K9_??x@K9_??x@!K9_??x@*      ??!       2      ??!       :	ù????ù????!ù????B      ??!       J	GY???.??GY???.??!GY???.??R      ??!       Z	GY???.??GY???.??!GY???.??b      ??!       JGPUY.?H?3&@b q?i??ҭ:@y?b??$O@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputg&?9y???!g&?9y???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?հ???!B?7?????0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??U+???!؀?é???"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad?e,U??!?
i4???"1
model/Conv1D_3/conv1dConv2D?@?t???!??B?m??"{
\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?*?A n??!?u?ƺ??"\
=model/Conv1D_2/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?*?A n??!Z@?????"}
^gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??ߑ?3??!o???O??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@?S????!?W?????0"1
model/Conv1D_2/conv1dConv2D?Z?????!j?aw????Q      Y@Y&W?+?)@a?????U@q???mk?&@y?o??Z??"?
both?Your program is MODERATELY input-bound because 11.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?24.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 