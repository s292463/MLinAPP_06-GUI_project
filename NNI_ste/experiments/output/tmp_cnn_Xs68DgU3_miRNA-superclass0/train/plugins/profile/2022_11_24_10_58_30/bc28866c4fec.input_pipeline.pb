	c?: ?f @c?: ?f @!c?: ?f @	??킗?@??킗?@!??킗?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLc?: ?f @?M4?s??1t}?@A?ݰmQf??I?l??}??Y-{؜???rEagerKernelExecute 0*	
ףp=?c@2F
Iterator::Model?r.?Ue??!?NQ??H@)}?E֪?1????@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh?4?;??!??h??{A@)R??/Ie??1k????X@@:Preprocessing2U
Iterator::Model::ParallelMapV2nē?????!i??-@)nē?????1i??-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?g???c??!:??s??&@)Yk(?ц?1?
E"vB@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????$?{?!??AŦJ@)????$?{?1??AŦJ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??N]???!S??Z4?I@)G?˵hz?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapC??À??! ??8EB@)?M???Pd?1X?k?)??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?????_?!JB?ד???)?????_?1JB?ד???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???=?Z?!?R=????)???=?Z?1?R=????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?24.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??킗?@I???=%C@Q"u9y	?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?M4?s???M4?s??!?M4?s??      ??!       "	t}?@t}?@!t}?@*      ??!       2	?ݰmQf???ݰmQf??!?ݰmQf??:	?l??}???l??}??!?l??}??B      ??!       J	-{؜???-{؜???!-{؜???R      ??!       Z	-{؜???-{؜???!-{؜???b      ??!       JGPUY??킗?@b q???=%C@y"u9y	?M@