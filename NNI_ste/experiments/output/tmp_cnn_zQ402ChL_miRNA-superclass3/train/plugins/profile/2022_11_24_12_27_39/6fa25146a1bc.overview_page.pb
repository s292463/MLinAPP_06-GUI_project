? 	:??KT?@:??KT?@!:??KT?@	?g?s?m???g?s?m??!?g?s?m??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL:??KT?@v???_?@13???/??A?p?;??I???jdW@Y??~?7??rEagerKernelExecute 0*	?"?????@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?
?.H??!????H@)㪲?????1H;O??F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k?????!R	-P??D@)??c?g^??1??sPC@:Preprocessing2F
Iterator::ModelxB???ϱ?!??+?@@)6?e?Ԩ?1??&???@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??1z??!?w?C<?@)Ӿ??z??1?fX??@:Preprocessing2U
Iterator::Model::ParallelMapV2u/3l???!\X?Y^b??)u/3l???1\X?Y^b??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ek}???!?Kt???)p\?M4??1???G~>??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??
???!????$???)TpxADj??1c?ʖ????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchx?W?L??!U?l?6???)x?W?L??1U?l?6???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?7?06??!?*??"F@)2t??ׁ?1r?R?J??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????z?!?*??f???)????z?1?*??f???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice\>???v?!?{z???)\>???v?1?{z???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeZ?rL?o?!&?{\G???)Z?rL?o?1&?{\G???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?X?? ~?!t#?	Rm??)j???]?1?'	T^??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?mO???N?!???7`??)?mO???N?1???7`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?g?s?m??I@?	2?Q@Q?x?A_`:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v???_?@v???_?@!v???_?@      ??!       "	3???/??3???/??!3???/??*      ??!       2	?p?;???p?;??!?p?;??:	???jdW@???jdW@!???jdW@B      ??!       J	??~?7????~?7??!??~?7??R      ??!       Z	??~?7????~?7??!??~?7??b      ??!       JGPUY?g?s?m??b q@?	2?Q@y?x?A_`:@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter]?.?????!]?.?????0"1
model/Conv1D_2/conv1dConv2D??	?.??!?@|?e??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?0?????!??????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???r-??!???!??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??????!?Y??I??0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?S?H?q??!wn?????"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose)XJ???!????
??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?/O?"??!?????N??0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterG?????!????ς??0"3
model/Conv1D_1/BiasAddBiasAdd"??9?J??!?9?!???Q      Y@YQ^Cye4@al(????S@q?:?.?82@y???f?9??"?
both?Your program is POTENTIALLY input-bound because 30.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?41.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 