	u?Hg`d!@u?Hg`d!@!u?Hg`d!@	SF?',@SF?',@!SF?',@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLu?Hg`d!@BZc?	???1??
??@A??"[As?Ii??? @Y:??q????rEagerKernelExecute 0*	V-P?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK?Բ???!????#S@)w?T????1???R@:Preprocessing2F
Iterator::Model???)x??!???p?-@)o??ʡ??1{l/(w?#@:Preprocessing2U
Iterator::Model::ParallelMapV2?????!E???g@)?????1E???g@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD??)X???! ?_?qAU@)f?c]?F??1?ْ?ݲ	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???'?.??!H???@)
??t??134Em<?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice[#?qp???!?[?Z? @)[#?qp???1?[?Z? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{??????!hy??"??){??????1hy??"??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'g(?x???!Sڠ\4@)?c?3?%k?1ޑ\ţ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?23.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9SF?',@I??q'`:@QV_9?$?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	BZc?	???BZc?	???!BZc?	???      ??!       "	??
??@??
??@!??
??@*      ??!       2	??"[As???"[As?!??"[As?:	i??? @i??? @!i??? @B      ??!       J	:??q????:??q????!:??q????R      ??!       Z	:??q????:??q????!:??q????b      ??!       JGPUYSF?',@b q??q'`:@yV_9?$?M@