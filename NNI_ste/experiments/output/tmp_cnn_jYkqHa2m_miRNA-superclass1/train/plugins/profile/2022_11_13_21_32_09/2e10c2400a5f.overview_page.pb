?	\?	?M@\?	?M@!\?	?M@	?R?)?@?R?)?@!?R?)?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL\?	?M@g??)??1)v4?@A?[?~l?_?Iz ???!??YWC?K??rEagerKernelExecute 0*		?Zda}@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZ??լ3??!Q????rR@)?k????1?|????Q@:Preprocessing2F
Iterator::Model??k????!?zh?k-@)x??#????1+?)??#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^?Y-?Ǥ?!y?wD!@)?St$????1Ѱ??@@:Preprocessing2U
Iterator::Model::ParallelMapV2'i??֖?!?!??N?@)'i??֖?1?!??N?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?|x??!D?`?@)??_?|x??1D?`?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??v????!???2?RU@)???q????1??lj?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|,G?@~?!???#??)|,G?@~?1???#??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??;3?p??!\P??R@)k*??.?n?1???B?`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?22.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?R?)?@I???i???@QdJ????O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	g??)??g??)??!g??)??      ??!       "	)v4?@)v4?@!)v4?@*      ??!       2	?[?~l?_??[?~l?_?!?[?~l?_?:	z ???!??z ???!??!z ???!??B      ??!       J	WC?K??WC?K??!WC?K??R      ??!       Z	WC?K??WC?K??!WC?K??b      ??!       JGPUY?R?)?@b q???i???@ydJ????O@?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd??m?Ұ?!d??m?Ұ?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~><yvb??!R?NU	??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput???S??!???\??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?^3?!ê?!?|??????0"1
model/Conv1D_2/conv1dConv2D????wϩ?!P?ia???"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??Vn???!pHI,e???"1
model/Conv1D_3/conv1dConv2D@?Mƺ??!???|v??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradD@EٻԢ?!??; ???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterk׿*ԡ?!͟?????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad7??????!>#?=t???Q      Y@Y      )@a     ?U@q???6??4@y???u?@??"?
both?Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?22.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 