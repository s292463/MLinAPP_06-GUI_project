	zpw?n+@zpw?n+@!zpw?n+@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCzpw?n+@?1???? @1?c?3??@A?-?\o??I????b?@rEagerKernelExecute 0*	??/?c@2F
Iterator::ModelF^????!$???.bH@)??~?n??1X?3?nV?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?|?q ??!=I?s??:@)?A_z?s??1??%??b6@:Preprocessing2U
Iterator::Model::ParallelMapV26\?-??!??M??m1@)6\?-??1??M??m1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?q??????!??%p0@)H?3?9A??1T??\?z!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*X?l:??!na,?;?@)*X?l:??1na,?;?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???H????!?*?RѝI@)?[<?????1?&?g?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??A%?c|?!k????4@)??A%?c|?1k????4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?"????!Â?PH2@)r?Md?g?1?+j????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?39.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????P@Q???? v@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1???? @?1???? @!?1???? @      ??!       "	?c?3??@?c?3??@!?c?3??@*      ??!       2	?-?\o???-?\o??!?-?\o??:	????b?@????b?@!????b?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????P@y???? v@@