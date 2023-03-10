?	??f?R? @??f?R? @!??f?R? @	QU???@QU???@!QU???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??f?R? @÷?n????1??nJ?@A??X???r?I??y?)z@Yyt#,*???rEagerKernelExecute 0*	j?t?Te@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatB>?٬???!?鷖?@@)????K7??1?P?J??<@:Preprocessing2F
Iterator::Model??ͪ?ղ?!?ج???E@)(֩?=#??1{?G??<@:Preprocessing2U
Iterator::Model::ParallelMapV2???????!ҙy k?,@)???????1ҙy k?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??|?rٸ?!N'SqL@)?P1?߄??1?Mȡ@2%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??]gE??!2???3@)??]gE??12???3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatem ]lZ)??!????'@)Cp\?M??1??^-j?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%u?~?!?r???:@)%u?~?1?r???:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt
??????!?????;*@)<P?<?f?1?H)B??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?28.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9QU???@IhM-?9B@Q????KM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	÷?n????÷?n????!÷?n????      ??!       "	??nJ?@??nJ?@!??nJ?@*      ??!       2	??X???r???X???r?!??X???r?:	??y?)z@??y?)z@!??y?)z@B      ??!       J	yt#,*???yt#,*???!yt#,*???R      ??!       Z	yt#,*???yt#,*???!yt#,*???b      ??!       JGPUYQU???@b qhM-?9B@y????KM@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputk??a??!k??a??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterQ?B+M???!
I=?????0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad??????!?Ʋ??8??"1
model/Conv1D_3/conv1dConv2D?X?#????!l??P????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad@?^???!|?\???"\
=model/Conv1D_2/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeQ?h????!???7??"}
^gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?ˠ???!熷?????"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?78}???!??^m1L??0"{
\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose0?7???!h?%n2???"1
model/Conv1D_2/conv1dConv2D??,?n??!"?(???Q      Y@YD+l$Z)@a?z2~??U@q???*?t2@y?9x3;??"?
both?Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?28.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 