	d??u @d??u @!d??u @	??a.??3@??a.??3@!??a.??3@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLd??u @???{,??1??	/????A?,??\n??I??k~?e@Y?/h!???rEagerKernelExecute 0*	^?I+c@2F
Iterator::ModelHj?drj??!+?S$ͺH@)?a??4???1!,?7??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[?7?qç?!??o?iD>@)#???R??1?VS??9@:Preprocessing2U
Iterator::Model::ParallelMapV2X?B?_˛?!5?{?b?1@)X?B?_˛?15?{?b?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??f?|??!??A
X^@)??f?|??1??A
X^@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Dh׳?!?;??2EI@){Cr??1eU??~@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?B:<????!?VS??f)@)E?4f??1??dUo@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?y?Cn?{?!?&c?z?@)?y?Cn?{?1?&c?z?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??s?f???!?Q'???,@)Ih˹We?1?֟?C.??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 19.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?33.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t24.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??a.??3@Ioq?)?+M@Q6YU~??5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???{,?????{,??!???{,??      ??!       "	??	/??????	/????!??	/????*      ??!       2	?,??\n???,??\n??!?,??\n??:	??k~?e@??k~?e@!??k~?e@B      ??!       J	?/h!????/h!???!?/h!???R      ??!       Z	?/h!????/h!???!?/h!???b      ??!       JGPUY??a.??3@b qoq?)?+M@y6YU~??5@