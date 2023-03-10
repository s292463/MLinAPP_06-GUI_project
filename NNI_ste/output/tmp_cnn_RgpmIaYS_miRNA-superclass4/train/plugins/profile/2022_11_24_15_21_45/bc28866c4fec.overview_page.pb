?	6?$#g?4@6?$#g?4@!6?$#g?4@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC6?$#g?4@?>+N??1ˁj??0@A??x?'??Ix{????rEagerKernelExecute 0*	?O??nd@2F
Iterator::Model???`?$??!lS??z?I@)~;??"??13?x???C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?[ A?c??!C?Y?8@)=a??M??1???7@?4@:Preprocessing2U
Iterator::Model::ParallelMapV2?1=a???!?0?y7N(@)?1=a???1?0?y7N(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?T??????!??$3?'@)?T??????1??$3?'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateyY??!?#??1@)???Հ?1Y?sdm@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipߤiP4??!??z?WH@)??Cl??1-e???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorԷ???x?!D]?y??@)Է???x?1D]?y??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Pi??>??!?h??;?2@)ǁW˝i?1?ǚ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIt??/?1@Q#M9??T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?>+N???>+N??!?>+N??      ??!       "	ˁj??0@ˁj??0@!ˁj??0@*      ??!       2	??x?'????x?'??!??x?'??:	x{????x{????!x{????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qt??/?1@y#M9??T@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???????!???????0"1
model/Conv1D_2/conv1dConv2D??C`????!?Bf?s???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?<?ۗ5??!?Ir????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????د?!2?0????0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??:???!?SPq????0"1
model/Conv1D_3/conv1dConv2D?ح3/???!??Wa???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad9A\ᷤ??!???լ???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradz??????!??].f??"3
model/Conv1D_1/BiasAddBiasAdd????O??!??G'???"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?O??ޕ?!?n?)??Q      Y@Y???cj`'@a?O???V@qڼ?5@y0\C2H͋?"?
both?Your program is POTENTIALLY input-bound because 8.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?21.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 