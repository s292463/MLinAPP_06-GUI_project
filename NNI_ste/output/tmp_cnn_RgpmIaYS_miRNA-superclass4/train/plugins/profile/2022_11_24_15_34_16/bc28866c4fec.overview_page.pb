?	???]M?-@???]M?-@!???]M?-@	v ???w??v ???w??!v ???w??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???]M?-@?8Q????1? :v?&@A???v?>??I?_??D??Y??:M??rEagerKernelExecute 0*	????x?f@2F
Iterator::Model?-y<-??!?~r??	I@)?Ky ???1??AN	B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat~??$????!?rYA2?7@)?Y?X??1b??I?3@:Preprocessing2U
Iterator::Model::ParallelMapV2?C p???!Q?N?J,@)?C p???1Q?N?J,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?sCSv??!??-???3@){K9_콘?1G??|?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicee5]Ot]??!?gg?ER@)e5]Ot]??1?gg?ER@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;Qi??!d??_?H@)$?`S?Q??1f??I??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?K?A??~?!P?!i??@)?K?A??~?1P?!i??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap*????!9S??5@)??)t^cg?1?JS?D??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9v ???w??I??m??c5@QY$U?7AS@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8Q?????8Q????!?8Q????      ??!       "	? :v?&@? :v?&@!? :v?&@*      ??!       2	???v?>?????v?>??!???v?>??:	?_??D???_??D??!?_??D??B      ??!       J	??:M????:M??!??:M??R      ??!       Z	??:M????:M??!??:M??b      ??!       JGPUYv ???w??b q??m??c5@yY$U?7AS@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter_z?%???!_z?%???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?m?&??!>???????0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??kK??!??E???0"1
model/Conv1D_3/conv1dConv2D??????!?'??qi??"1
model/Conv1D_2/conv1dConv2D?-,,[???!??M*d2??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??3B?M??!??????0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradݥ?Ϣ?!X┓6??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad m??[???!*?^Pw.??"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeH????Z??!?di&???"3
model/Conv1D_1/BiasAddBiasAdd?s???W??!???ڥ???Q      Y@YƖ???&@a'??d@=V@q?-đ?27@y?A??Y??"?
both?Your program is POTENTIALLY input-bound because 11.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?23.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 