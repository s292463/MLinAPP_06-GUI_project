	?Q?(?sM@?Q?(?sM@!?Q?(?sM@	AKe?????AKe?????!AKe?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?Q?(?sM@??ŦU@1???I??A???mR??IoF?W?#I@Y2???????rEagerKernelExecute 0*	?????d@2F
Iterator::Model??
}????!(?j`?(J@)5B?S?[??1R?,6A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???'?.??!V?V??8@)?1??????1?i/&?4@:Preprocessing2U
Iterator::Model::ParallelMapV2?1?Mc{??!2???1@)?1?Mc{??12???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??3?!GU???0@)????aN??1??MX-?#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?_?????!?
??@)?_?????1?
??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?:?????!???m?G@)?L?x$^~?1?w?<kn@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZI+???y?!XBg??|@)ZI+???y?1XBg??|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap0c
?8???!\ի?}?2@)?
?.?f?1F8?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?85.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9AKe?????I?A????X@Qm??8?^??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ŦU@??ŦU@!??ŦU@      ??!       "	???I?????I??!???I??*      ??!       2	???mR?????mR??!???mR??:	oF?W?#I@oF?W?#I@!oF?W?#I@B      ??!       J	2???????2???????!2???????R      ??!       Z	2???????2???????!2???????b      ??!       JGPUYAKe?????b q?A????X@ym??8?^??