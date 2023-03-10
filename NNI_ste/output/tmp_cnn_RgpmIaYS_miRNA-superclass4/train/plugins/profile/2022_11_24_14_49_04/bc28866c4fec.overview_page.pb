?	n/?)@n/?)@!n/?)@	??y]?@??y]?@!??y]?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLn/?)@j0?G???1_???:?!@A?s?v?4??I?a?????YS???t??rEagerKernelExecute 0*??n??f@)       =2F
Iterator::Model߈?Y?h??!	? ?G@)?5?o????1_*چ??;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\?O???!???m?X?@)???{??1??rk?y;@:Preprocessing2U
Iterator::Model::ParallelMapV2??ID???!?{Rn2@)??ID???1?{Rn2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????9??!Qx?	X?/@)*:??H??1??fF?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicer????)??!?@???I@)r????)??1?@???I@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZippUj???!???]?J@)L?$zł?1oi{T=@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?q?j??|?!J! ??@)?q?j??|?1J! ??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps????(??!?^???l1@)àL???h?1(b????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??y]?@I??????;@Q*
[.?vQ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	j0?G???j0?G???!j0?G???      ??!       "	_???:?!@_???:?!@!_???:?!@*      ??!       2	?s?v?4???s?v?4??!?s?v?4??:	?a??????a?????!?a?????B      ??!       J	S???t??S???t??!S???t??R      ??!       Z	S???t??S???t??!S???t??b      ??!       JGPUY??y]?@b q??????;@y*
[.?vQ@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter*y?'_??!*y?'_??0"1
model/Conv1D_2/conv1dConv2D?ބ?<???!j?;$ƨ??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput????!|??Ez??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??u????!?s?xg???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad???w^???!??P????"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?^?0???!?h?|T??0"1
model/Conv1D_3/conv1dConv2D?t?????!Jr??m???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad:D??y???!???,?t??"3
model/Conv1D_1/BiasAddBiasAdd#h???ˡ?!??g?1???"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose???8?ɡ?!&??i???Q      Y@Y??????'@aV@q??":?1@yy??뇏??"?
both?Your program is POTENTIALLY input-bound because 13.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?17.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 