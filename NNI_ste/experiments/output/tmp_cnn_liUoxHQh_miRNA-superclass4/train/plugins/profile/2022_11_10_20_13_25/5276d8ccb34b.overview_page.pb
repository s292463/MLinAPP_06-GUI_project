?	???jd?"@???jd?"@!???jd?"@	_2?c???_2?c???!_2?c???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???jd?"@?.???ur?1VIddY@A?;?2T???IE/?Xn?@Y?nJy???rEagerKernelExecute 0*	%??Ca@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????K??!nE4???@)???R???1?٘?;@:Preprocessing2F
Iterator::Model?O?????!?K??-'F@)??ĬC??1j	?b?8@:Preprocessing2U
Iterator::Model::ParallelMapV2?m?s??!??)??3@)?m?s??1??)??3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???'???!?v??>"@)???'???1?v??>"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?"1?0??!Qs9??/@)"?^F?܂?1q?>?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????y??!a?*??K@) ?M?????1OL]?a7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM?~2Ƈy?!?<?mf@@)M?~2Ƈy?1?<?mf@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????!9??!??ª-2@);ŪAh?1Zo?qDW@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?55.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9`2?c???II}???R@Q???/?X8@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?.???ur??.???ur?!?.???ur?      ??!       "	VIddY@VIddY@!VIddY@*      ??!       2	?;?2T????;?2T???!?;?2T???:	E/?Xn?@E/?Xn?@!E/?Xn?@B      ??!       J	?nJy????nJy???!?nJy???R      ??!       Z	?nJy????nJy???!?nJy???b      ??!       JGPUY`2?c???b qI}???R@y???/?X8@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterׂ,?,??!ׂ,?,??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput???O????!??v?O??0"1
model/Conv1D_2/conv1dConv2DnE\?I??!2?R-z??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter&>A	h???!??BSX??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?u!?ġ?!t?q׭???0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradج˪?N??!?ʌ????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad`ˡ8??!?U@?z??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad`ȴE???!Kd@??C??"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits?h??#i??!?
??Y???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter3v??0??!8?+?Z}??0Q      Y@Y|??'@a|??V@q??p?.LD@y4??x??"?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?55.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 