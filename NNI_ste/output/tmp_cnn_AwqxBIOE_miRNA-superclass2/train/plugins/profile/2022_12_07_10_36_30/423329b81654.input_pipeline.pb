	?L??+@?L??+@!?L??+@	M'?2?@M'?2?@!M'?2?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?L??+@??v1?t??1U3k) ?@A??9??w?I(D?!T	 @YH?'????rEagerKernelExecute 0*??ʡ?f@)       =2F
Iterator::Model^*6?uĹ?!'?`??K@)Q?l???1?zErAC@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????8??!?K??*<@)?s??˦?19?e?}8@:Preprocessing2U
Iterator::Model::ParallelMapV26?$#ga??!~|6J9?0@)6?$#ga??1~|6J9?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice? x|{??! ?	?B:@)? x|{??1 ?	?B:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJV?????!k1C#~&@)?m?2d??1?lX??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?#+?ƴ?!?F??IQF@)?̯? ?|?1?n?G?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensore????`{?!?p??i@)e????`{?1?p??i@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB{???w??!?g~?Q6)@)????(@d?1?Bj?r???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?26.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9M'?2?@I?N?E׸?@Q?q??3P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??v1?t????v1?t??!??v1?t??      ??!       "	U3k) ?@U3k) ?@!U3k) ?@*      ??!       2	??9??w???9??w?!??9??w?:	(D?!T	 @(D?!T	 @!(D?!T	 @B      ??!       J	H?'????H?'????!H?'????R      ??!       Z	H?'????H?'????!H?'????b      ??!       JGPUYM'?2?@b q?N?E׸?@y?q??3P@