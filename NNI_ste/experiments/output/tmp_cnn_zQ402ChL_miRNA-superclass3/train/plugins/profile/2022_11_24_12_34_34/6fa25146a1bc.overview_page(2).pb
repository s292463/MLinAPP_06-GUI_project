?	/?o?S?y@/?o?S?y@!/?o?S?y@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:/?o?S?y@be4?y???1g????)w@I?)V?`E@rEagerKernelExecute 0*	?p=
?f@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate????o??!?LBo~E@)?JC?B??1????G1D@:Preprocessing2F
Iterator::Model2s??cͰ?!8?R"??B@)q??H/j??18?d???9@:Preprocessing2U
Iterator::Model::ParallelMapV2???0a??!sz?PY?&@)???0a??1sz?PY?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?XP?i??!?a??_kO@)??el?f??1??6??\!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?B ?8???!??f?Z?"@)2q? ???1?؝?!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'*?Tv?!x$???@)?'*?Tv?1x$???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????(@??!)ư?dF@)?e??
j?1??=?m???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor_~?Ɍ?e?!?F???)_~?Ɍ?e?1?F???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?????_?!8??????)?????_?18??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIo???$@Q??ΣgV@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	be4?y???be4?y???!be4?y???      ??!       "	g????)w@g????)w@!g????)w@*      ??!       2      ??!       :	?)V?`E@?)V?`E@!?)V?`E@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qo???$@y??ΣgV@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?S????!?S????0"1
model/Conv1D_2/conv1dConv2Do|=?????!?]Hx????"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputګ?E????!??8?P??0"1
model/Conv1D_3/conv1dConv2D?jC???!T????P??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?	??p???!???N???0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterl[?ڠy??!}E?Uܨ??0"1
model/Conv1D_4/conv1dConv2DQ?O????!???s?E??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?[?????!`0H糵??0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??ʈh??!?t
V??0"1
model/Conv1D_1/conv1dConv2D?ݮ7\???!?R{/~??Q      Y@Y?{?1m@aD?,??W@q Tf?2?K@yz??J^R?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?55.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 