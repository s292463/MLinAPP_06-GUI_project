?	\䞮?!@\䞮?!@!\䞮?!@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC\䞮?!@??۟???1@?ϝ`?@A5&?\R??I??jׄ?@rEagerKernelExecute 0*	bX9? v@2U
Iterator::Model::ParallelMapV2U3k) ???!Q?Sc?J@)U3k) ???1Q?Sc?J@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat:?m½2??!?	?O1@)R?y9쾫?1??0?W?.@:Preprocessing2F
Iterator::Modelt}???!??????P@)|(ђ?Ӫ?1ڢG?m?-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateD??{???!?r?8#&@)uۈ'???1??"??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice&VF#?W??!ƠH??r@)&VF#?W??1ƠH??r@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGˁjۼ?!?????@@)????w??18?Q??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA?G??{?!???Xk???)A?G??{?1???Xk???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV?Z???!ȀMa|(@)!???'*k?1?᠖E$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?33.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI =?'J@Q????3?G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??۟?????۟???!??۟???      ??!       "	@?ϝ`?@@?ϝ`?@!@?ϝ`?@*      ??!       2	5&?\R??5&?\R??!5&?\R??:	??jׄ?@??jׄ?@!??jׄ?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q =?'J@y????3?G@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??1>?u??!??1>?u??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?	?? ɾ?!:?V????0"1
model/Conv1D_2/conv1dConv2D??q???!|׸d?p??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?u^	?a??!?t?j???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput????g??!k&ƽK???0"1
model/Conv1D_3/conv1dConv2D??.?HI??!YyF????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?&?UX???!a???ZB??"1
model/Conv1D_4/conv1dConv2Do?'0????!t?k߈??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad"?? u??!???1???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<?|m??!???????0Q      Y@Y      )@a     ?U@q??&_9F@y??̤k???"?
both?Your program is POTENTIALLY input-bound because 18.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?33.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?44.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 