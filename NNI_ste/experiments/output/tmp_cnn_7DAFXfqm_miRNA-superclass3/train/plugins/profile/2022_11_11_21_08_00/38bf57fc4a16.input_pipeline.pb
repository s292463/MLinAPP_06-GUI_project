	??rf?@??rf?@!??rf?@	?w'8#@?w'8#@!?w'8#@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??rf?@i????1\sG??@An?2d???IT? ?!'@Y??I`s??rEagerKernelExecute 0*	T㥛?(c@2F
Iterator::Model??[[??!x?F@)?IӠh??1?HJ?ݻ>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??0????!P???7@)߿yq⫝?1?ښ?2@:Preprocessing2U
Iterator::Model::ParallelMapV25??-</??!?~?˪?*@)5??-</??1?~?˪?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??o?????!??ۼS$)@)??o?????1??ۼS$)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*?~?????!????f?K@)Q???J???1??FF?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatez4Փ?G??!???)?3@)???????1?p???o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)w????y?!ȝ?k@))w????y?1ȝ?k@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapWBwI???!???j)?5@)?????g?1?p???o??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?w'8#@It1??#7L@QW???vD@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	i????i????!i????      ??!       "	\sG??@\sG??@!\sG??@*      ??!       2	n?2d???n?2d???!n?2d???:	T? ?!'@T? ?!'@!T? ?!'@B      ??!       J	??I`s????I`s??!??I`s??R      ??!       Z	??I`s????I`s??!??I`s??b      ??!       JGPUY?w'8#@b qt1??#7L@yW???vD@