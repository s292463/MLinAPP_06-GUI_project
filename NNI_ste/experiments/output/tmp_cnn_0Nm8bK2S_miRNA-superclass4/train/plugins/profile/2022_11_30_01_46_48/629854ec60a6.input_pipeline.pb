	&??[XC3@&??[XC3@!&??[XC3@	?? @???? @??!?? @??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC&??[XC3@ĕ?wF???1??Ӝ?|0@I???G<??Y0?AC???rEagerKernelExecute 0*	rh??|9u@2F
Iterator::Model(*?T??!????5?R@)??N^??1A?R??PP@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat${??!U??!(?N?K0@)?1uWv???1~?ao?y,@:Preprocessing2U
Iterator::Model::ParallelMapV2*?#??t??!??YP?!@)*?#??t??1??YP?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceo?ŏ1??!۸K?ҭ
@)o?ŏ1??1۸K?ҭ
@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate6Φ#????!m?Ľ?@)?I?p??1/!???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.???=???!???m)?9@)????????1?I?Rr?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorIJzZ?|?!EgR?u @)IJzZ?|?1EgR?u @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{K9_콘?!?\R?u@)-???ai?1Y`?m???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?? @??IX%??i)@Q??eU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ĕ?wF???ĕ?wF???!ĕ?wF???      ??!       "	??Ӝ?|0@??Ӝ?|0@!??Ӝ?|0@*      ??!       2      ??!       :	???G<?????G<??!???G<??B      ??!       J	0?AC???0?AC???!0?AC???R      ??!       Z	0?AC???0?AC???!0?AC???b      ??!       JGPUY?? @??b qX%??i)@y??eU@