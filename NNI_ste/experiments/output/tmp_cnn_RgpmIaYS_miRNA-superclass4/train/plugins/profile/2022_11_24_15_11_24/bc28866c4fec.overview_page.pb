?	?f?|?D)@?f?|?D)@!?f?|?D)@	ޱ????ޱ????!ޱ????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?f?|?D)@`r??Z?@1?}V?)U @A/??ؗ?I? m?Y???YK %vmo??rEagerKernelExecute 0*	D?l???d@2F
Iterator::Model̶?ֈ`??!d(L?b?G@)'???K??1???l??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?@I?0??!?]?sk?8@)Y?O0???1A???4@:Preprocessing2U
Iterator::Model::ParallelMapV2????M??!;??aY0@)????M??1;??aY0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??b?D??!??A?#?2@)?;? є?1?????{(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS?A?Ѫ??!W+A???@)S?A?Ѫ??1W+A???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)H4?"??!?׳J?J@)?5?e܄?1?q??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????|?!?r ?1?@)?????|?1?r ?1?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??u?ݡ?!?????5@)??2R??l?1??a:? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?13.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ޱ????IF?֕??@@Q?G???(P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`r??Z?@`r??Z?@!`r??Z?@      ??!       "	?}V?)U @?}V?)U @!?}V?)U @*      ??!       2	/??ؗ?/??ؗ?!/??ؗ?:	? m?Y???? m?Y???!? m?Y???B      ??!       J	K %vmo??K %vmo??!K %vmo??R      ??!       Z	K %vmo??K %vmo??!K %vmo??b      ??!       JGPUYޱ????b qF?֕??@@y?G???(P@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????k???!????k???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradirU??ڪ?!u??O???"1
model/Conv1D_2/conv1dConv2Dw|Gu???!??߬????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad0Ş?Pޥ?!?P?Єq??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputx~?Iݥ?!>x??kt??0"3
model/Conv1D_1/BiasAddBiasAdd???ݺp??!?uV?????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose????Z??!n???M??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose????U??!???愸??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose???=a??!???N???"-
model/Conv1D_1/ReluRelu#?d?????!;5AD??Q      Y@Y:??s?9'@ac?1?V@q????)3@y>??T???"?
both?Your program is POTENTIALLY input-bound because 19.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?19.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 