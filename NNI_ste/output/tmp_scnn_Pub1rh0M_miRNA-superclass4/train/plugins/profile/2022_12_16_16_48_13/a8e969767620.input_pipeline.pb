	?+???NE@?+???NE@!?+???NE@	&"?9 @&"?9 @!&"?9 @"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?+???NE@'i??? @1?????<@I?$[]Nq @Y?NR@r0*	#??~ja@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????n???!???A@)??e?c]??1?ak?)=@:Preprocessing2U
Iterator::Model::ParallelMapV2??ǁW??!CD<?œ3@)??ǁW??1CD<?œ3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap&r?????!a&?J?2;@)]?????1w~?a?/@:Preprocessing2F
Iterator::Model?6o????!?3Ѩ\?@)?p?q?t??1@?)竐'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??
E???!ME?ư?&@)??
E???1ME?ư?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip|?Y?H???!???(Q@)?$?@??1q?T}??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?,^,??!4y??i@)?,^,??14y??i@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?19.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s4.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9&"?9 @I?,?_5;8@Q??wK?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'i??? @'i??? @!'i??? @      ??!       "	?????<@?????<@!?????<@*      ??!       2      ??!       :	?$[]Nq @?$[]Nq @!?$[]Nq @B      ??!       J	?NR@?NR@!?NR@R      ??!       Z	?NR@?NR@!?NR@b      ??!       JGPUY&"?9 @b q?,?_5;8@y??wK?P@