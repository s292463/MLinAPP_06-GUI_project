	5s???@5s???@!5s???@	B??ɴk@B??ɴk@!B??ɴk@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL5s???@
??O?m??1=+i?7?@A0?????I\?W zr@Y???????rEagerKernelExecute 0*	d;?O???@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatecd?˻??!?	??S@)?? ?6q??1?rU	?R@:Preprocessing2F
Iterator::Model%?z?ۡ??!?0?n*@)??};???1?BwhF%"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ݯ|???!-???k4@)?????P??1WxN??@:Preprocessing2U
Iterator::Model::ParallelMapV2X??"?t??!???~??@)X??"?t??1???~??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????G???!`J?[?@)????G???1`J?[?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF?Swe??!???.r?U@)>?hɓ?1?1?:D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?b?=yx?![c}?r??)?b?=yx?1[c}?r??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?׃I????!?{??	0S@)??9]k?1lT?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?53.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9C??ɴk@I??	tK@Qh?Y?,5E@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
??O?m??
??O?m??!
??O?m??      ??!       "	=+i?7?@=+i?7?@!=+i?7?@*      ??!       2	0?????0?????!0?????:	\?W zr@\?W zr@!\?W zr@B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JGPUYC??ɴk@b q??	tK@yh?Y?,5E@