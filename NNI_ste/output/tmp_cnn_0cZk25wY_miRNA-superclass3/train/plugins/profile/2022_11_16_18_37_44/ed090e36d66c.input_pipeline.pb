	?%VF#?!@?%VF#?!@!?%VF#?!@	zwJ??@zwJ??@!zwJ??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?%VF#?!@?c?3?%??1ض(?A?@A?!T????IaS?Q????Y)??q??rEagerKernelExecute 0*	??Mb?u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??l????!?R??w?I@)???!?k??1p??k?G@:Preprocessing2F
Iterator::Modeld???^D??!`????>@)??
?|$??1?\?x??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatAI?0e??!X??r?+@)????G???1ܹ?r?b'@:Preprocessing2U
Iterator::Model::ParallelMapV2??[v???!`Z?5D?@)??[v???1`Z?5D?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice&??????!=r??i$@)&??????1=r??i$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM??y ???!(C?HQ@)??0Bx??1$nj?+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???C?}?!?5??? @)???C?}?1?5??? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|a2U0*??!؟???7J@)??c?M*j?1?GS?K???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?18.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9zwJ??@I?	9?$A@Q???N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c?3?%???c?3?%??!?c?3?%??      ??!       "	ض(?A?@ض(?A?@!ض(?A?@*      ??!       2	?!T?????!T????!?!T????:	aS?Q????aS?Q????!aS?Q????B      ??!       J	)??q??)??q??!)??q??R      ??!       Z	)??q??)??q??!)??q??b      ??!       JGPUYzwJ??@b q?	9?$A@y???N@