	<??k $@<??k $@!<??k $@	?:g?C@?:g?C@!?:g?C@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC<??k $@???????1???=?@@I`"?:??@YqǛ????rEagerKernelExecute 0*	?G?zXv@2U
Iterator::Model::ParallelMapV2Bz?"n??!?B?J@)Bz?"n??1?B?J@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?]J]2???!?A|4.3@)?#??????1???<?0@:Preprocessing2F
Iterator::ModelX?????!b??*?P@)W$&??[??1??%?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceo?[t???!Kт???@)o?[t???1Kт???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(?.??|??!<OVګ?@@)2t???1H???}}@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatecD?в???!??`??q!@)??_?|x??1~8~???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?_??s??!??$???@)?_??s??1??$???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape6?$#g??!I?ꭇ#@)9CqǛ?f?10??8R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?:g?C@I?=?s?v:@Q?6?'5?Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "	???=?@@???=?@@!???=?@@*      ??!       2      ??!       :	`"?:??@`"?:??@!`"?:??@B      ??!       J	qǛ????qǛ????!qǛ????R      ??!       Z	qǛ????qǛ????!qǛ????b      ??!       JGPUY?:g?C@b q?=?s?v:@y?6?'5?Q@