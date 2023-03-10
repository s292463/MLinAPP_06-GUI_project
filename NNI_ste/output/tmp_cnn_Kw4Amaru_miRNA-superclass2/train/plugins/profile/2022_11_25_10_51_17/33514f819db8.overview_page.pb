?	s????%$@s????%$@!s????%$@	?{!?H?@?{!?H?@!?{!?H?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLs????%$@??C?X???1???ԱZ @A?O?mpf?I4???????Yy>?ͨ??rEagerKernelExecute 0*	6^?I?t@2F
Iterator::Modele4?y?S??!e???mR@)??r????1?Xf??pP@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatffffff??!֡?N?Z*@) |(Ѣ?19|&#?#&@:Preprocessing2U
Iterator::Model::ParallelMapV2?*2: 	??!??1?@)?*2: 	??1??1?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"H?V??!?k?XH:@)A???FX??1	?h???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatexԘsI??!H;???@)???c?3??1":Zf
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)狽_??!m<???@))狽_??1m<???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??2R??|?!s???? @)??2R??|?1s???? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapya?X5??!??Ŵ?{@)pA?,_g?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?12.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?{!?H?@I??P?_0@Q?1h??JT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??C?X?????C?X???!??C?X???      ??!       "	???ԱZ @???ԱZ @!???ԱZ @*      ??!       2	?O?mpf??O?mpf?!?O?mpf?:	4???????4???????!4???????B      ??!       J	y>?ͨ??y>?ͨ??!y>?ͨ??R      ??!       Z	y>?ͨ??y>?ͨ??!y>?ͨ??b      ??!       JGPUY?{!?H?@b q??P?_0@y?1h??JT@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?$"????!?$"????0"1
model/Conv1D_2/conv1dConv2D"K[??!X??>^a??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputeAσ?k??!??O??K??0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradq?5?
??!?:g????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?y?????!???????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??͟?! ?X"m???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose????u???!?W??????"3
model/Conv1D_1/BiasAddBiasAddؤ?y?k??!*"'Y????"-
model/Conv1D_1/ReluRelu?q"?l6??!HI?#o??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?g8W4.??!??\i?!??0Q      Y@Y?
*T?(@a?^?z??U@q??<]FB@y$????G??"?
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?12.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?36.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 