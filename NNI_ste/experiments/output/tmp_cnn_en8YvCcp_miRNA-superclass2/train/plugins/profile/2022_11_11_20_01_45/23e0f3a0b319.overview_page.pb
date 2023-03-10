?	??z???@??z???@!??z???@	#a????#a????!#a????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??z???@?q75???1A)Z?W??@A?6?h??m?I?d??E@YTƿϸp??rEagerKernelExecute 0*	K7?A`)`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate辜ٮЧ?!?b?+??A@)?zj?ե?1?p&R&~@@:Preprocessing2F
Iterator::Model??p???!f??O??E@)? |??1?d?? ?8@:Preprocessing2U
Iterator::Model::ParallelMapV2?2??Y??!3??od2@)?2??Y??13??od2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?fؑ?!t5?5??*@)3??????1)??.?"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipuF^?Ĳ?!?U?GZL@)?????P}?1?;EW$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?t_?lw?!???r?@)?t_?lw?1???r?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??x"????!???y??B@)C???-b?1|???u??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?`??_?!1??????)?`??_?11??????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor`x%?s}_?!ó?????)`x%?s}_?1ó?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9#a????IеzE@Q??/?MMW@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q75????q75???!?q75???      ??!       "	A)Z?W??@A)Z?W??@!A)Z?W??@*      ??!       2	?6?h??m??6?h??m?!?6?h??m?:	?d??E@?d??E@!?d??E@B      ??!       J	Tƿϸp??Tƿϸp??!Tƿϸp??R      ??!       Z	Tƿϸp??Tƿϸp??!Tƿϸp??b      ??!       JGPUY#a????b qеzE@y??/?MMW@?"1
model/Conv1D_2/conv1dConv2D?'?*s???!?'?*s???"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter"j?V?!??!?H??~??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput???????!???R????0"1
model/Conv1D_3/conv1dConv2DW?r?j??!G;3]???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput????j??!ἱ???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter_?7?N??!????~???0"1
model/Conv1D_4/conv1dConv2D$??n&??!e?????"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?O/Xك?!??l?|:??0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??g??ڂ?!"R?s????0"1
model/Conv1D_1/conv1dConv2Dl?c?0z?!In|H???Q      Y@Y
??%9@aX߇?m?W@q?VfA?/D@yû???=E?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 