	????/@????/@!????/@	??٥?????٥???!??٥???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????/@b?*?3??1$?&ݖ8@A[_$??\??I?J???* @YjK??`??rEagerKernelExecute 0*	}?5^??`@2F
Iterator::Model??[<????!??s?UH@)??D.8???1Z???c?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?+??f*??!*?y"`=@)rl=C8f??1R??9@:Preprocessing2U
Iterator::Model::ParallelMapV2??唀???!Y?ޥ-@)??唀???1Y?ޥ-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??҅?!?|??Rj@)??҅?1?|??Rj@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??X????!x7אݵ,@)??$????1S??Uh@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip74e?Ա?!N?$?$?I@)C?8
??1?\??J?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????t!v?!??{$??@)????t!v?1??{$??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J&?v??!G??L+0@)׆?q?&d?1?h?Fx??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??٥???I]???UE@Q?Y????K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b?*?3??b?*?3??!b?*?3??      ??!       "	$?&ݖ8@$?&ݖ8@!$?&ݖ8@*      ??!       2	[_$??\??[_$??\??![_$??\??:	?J???* @?J???* @!?J???* @B      ??!       J	jK??`??jK??`??!jK??`??R      ??!       Z	jK??`??jK??`??!jK??`??b      ??!       JGPUY??٥???b q]???UE@y?Y????K@