	(??h?\@(??h?\@!(??h?\@	L?L??y@L?L??y@!L?L??y@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL(??h?\@v??ť???1˟oK@AQ.?_x%??I?U???@Y4???l???rEagerKernelExecute 0*	?x?&1?r@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&p?n???!???,@M@)&o??????1?ۮ?'?I@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?T?G????!y/#?ߦ0@)^?SH??1?׭.B?,@:Preprocessing2F
Iterator::Model?z?G???!?mm=?H1@)`??橞?1zʫ?Q?#@:Preprocessing2U
Iterator::Model::ParallelMapV2b?qm???!	#^&m?@)b?qm???1	#^&m?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??}?????!W??0@)??}?????1W??0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6?.6???!????ޭT@)Y???RA??1K?JOU@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorе/??|?!]b???@)е/??|?1]b???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?+f????!?g\??M@)?j?=&Rj?1?&??|???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9M?L??y@I1???~#O@Q?:?Ꝛ?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v??ť???v??ť???!v??ť???      ??!       "	˟oK@˟oK@!˟oK@*      ??!       2	Q.?_x%??Q.?_x%??!Q.?_x%??:	?U???@?U???@!?U???@B      ??!       J	4???l???4???l???!4???l???R      ??!       Z	4???l???4???l???!4???l???b      ??!       JGPUYM?L??y@b q1???~#O@y?:?Ꝛ?@