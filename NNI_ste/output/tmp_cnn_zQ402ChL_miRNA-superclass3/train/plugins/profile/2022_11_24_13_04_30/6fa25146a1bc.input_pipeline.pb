	??T[8 @??T[8 @!??T[8 @		
.??x??	
.??x??!	
.??x??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??T[8 @?{)<h???1x???2??A<K?P???IX?vMH@Y??g?,??rEagerKernelExecute 0*	??C?lod@2F
Iterator::ModelL??T????!????0L@)0?x??n??1f????B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate`w???s??!?7&)??:@)s-Z??դ?1ӊ?s4?8@:Preprocessing2U
Iterator::Model::ParallelMapV2?JZ????!L 9?2@)?JZ????1L 9?2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??q?_??!???g??$@)V*?????1'`90B@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+?~NA??!t?m7?E@)?
?<??1F?l9e@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?r߉y?!??-т@)?r߉y?1??-т@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2r??ç?!O۩9d<@)*??% ?d?1??L???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?????K[?!Ԁ?N??)?????K[?1Ԁ?N??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?@?vX?!֜??X9??)?@?vX?1֜??X9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9
.??x??I?bGc?R@Q ?F?
8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?{)<h????{)<h???!?{)<h???      ??!       "	x???2??x???2??!x???2??*      ??!       2	<K?P???<K?P???!<K?P???:	X?vMH@X?vMH@!X?vMH@B      ??!       J	??g?,????g?,??!??g?,??R      ??!       Z	??g?,????g?,??!??g?,??b      ??!       JGPUY
.??x??b q?bGc?R@y ?F?
8@