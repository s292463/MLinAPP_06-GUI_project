	q???@q???@!q???@	ɱ??k@ɱ??k@!ɱ??k@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLq???@??T?:??1?2?68?@A??9x&4??I?'??????Yp??-??rEagerKernelExecute 0*	?rh??Td@2F
Iterator::Model????????!E@qj?I@)???"Ʈ?1ނ?,?yB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??D????!3c?8%9@)?Z?????1??L)5@:Preprocessing2U
Iterator::Model::ParallelMapV2]?????!????؏*@)]?????1????؏*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???(\???!?^??\I&@)???(\???1?^??\I&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy??M????!?????H@)??1????1??~??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???c[??!5n?0@)?D?u????1i?[T??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr??Q??z?!??{m??@)r??Q??z?1??{m??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap E??????!*N I?2@)?K?b?k?1?????? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?31.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t24.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9ʱ??k@IţX<??K@QHNӼB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??T?:????T?:??!??T?:??      ??!       "	?2?68?@?2?68?@!?2?68?@*      ??!       2	??9x&4????9x&4??!??9x&4??:	?'???????'??????!?'??????B      ??!       J	p??-??p??-??!p??-??R      ??!       Z	p??-??p??-??!p??-??b      ??!       JGPUYʱ??k@b qţX<??K@yHNӼB@