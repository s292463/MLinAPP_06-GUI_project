	$??X@$??X@!$??X@	?{Կ?@?{Կ?@!?{Կ?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL$??X@'???6??1?.??A?M?=????I7o??=@Y?X4????rEagerKernelExecute 0*	}?5^?5p@2U
Iterator::Model::ParallelMapV2>?>tA}??!u@LLWJ@)>?>tA}??1u@LLWJ@:Preprocessing2F
Iterator::ModelhY??????!?gFPX@Q@)??n???1H?<?R0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??*l???!?(7d?2@)??<???1??}?g/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???ȭI??!?䟂?@)???ȭI??1?䟂?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???B????!8Tu??w@)ؚ?????1??"U??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?>???!?a澞?>@)t?Lh?X??1??44??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?7??w??!}?I#??@)?7??w??1}?I#??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??3????!]?o??!@)??;??~f?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?49.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t20.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?{Կ?@Iܒ????Q@Q??&?m?6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'???6??'???6??!'???6??      ??!       "	?.???.??!?.??*      ??!       2	?M?=?????M?=????!?M?=????:	7o??=@7o??=@!7o??=@B      ??!       J	?X4?????X4????!?X4????R      ??!       Z	?X4?????X4????!?X4????b      ??!       JGPUY?{Կ?@b qܒ????Q@y??&?m?6@