	̳?V|K,@̳?V|K,@!̳?V|K,@	0Fj??@0Fj??@!0Fj??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL̳?V|K,@?V
?\???1XU/???&@A[???i??IU3k) ???Y</O????rEagerKernelExecute 0*	
ףp=&d@2F
Iterator::Model???????!=|>?e?G@)?G?`???1????sZ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!#?????![???#@@)?5?e?s??1??%4;@:Preprocessing2U
Iterator::Model::ParallelMapV2?5_%??!?J?Xn0@)?5_%??1?J?Xn0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????5"??!?W\??=@)????5"??1?W\??=@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip4GV~???!??J@)?m?2d??1?A?H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???`???!???_)@)zo???1x??D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Ss????!?vA??M@)??Ss????1?vA??M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???4???!~????,@)bJ$??(f?1/???v???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?14.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no90Fj??@I\a:V?|0@Qw?B|HT@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V
?\????V
?\???!?V
?\???      ??!       "	XU/???&@XU/???&@!XU/???&@*      ??!       2	[???i??[???i??![???i??:	U3k) ???U3k) ???!U3k) ???B      ??!       J	</O????</O????!</O????R      ??!       Z	</O????</O????!</O????b      ??!       JGPUY0Fj??@b q\a:V?|0@yw?B|HT@