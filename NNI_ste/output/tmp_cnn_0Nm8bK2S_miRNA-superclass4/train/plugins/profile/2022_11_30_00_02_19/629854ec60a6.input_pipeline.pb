	?׻-@?׻-@!?׻-@	?l\Rp?@?l\Rp?@!?l\Rp?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?׻-@3?뤾???1K??a'%@I?z3j?j @Y?X??+???rEagerKernelExecute 0*	z?&1?v@2F
Iterator::ModelX???<??!?# ??VR@)uu?b?T??1?6?k?"P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????#??!)?,V?+@)0c
?8???1B?j<?(@:Preprocessing2U
Iterator::Model::ParallelMapV2?T?????!?g?:?!@)?T?????1?g?:?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?뤾,???!%|?D@)?뤾,???1%|?D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?^????!ՍS???"@)?'??????1V?F?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?$A?
??!q?H?:@)?#?]J]??1??r2?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor˃?9D|?!Cg?`??)˃?9D|?1Cg?`??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??\??X??!?11t??$@)???$xCj?1?;?=????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?l\Rp?@IgV???6@Qu???7R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3?뤾???3?뤾???!3?뤾???      ??!       "	K??a'%@K??a'%@!K??a'%@*      ??!       2      ??!       :	?z3j?j @?z3j?j @!?z3j?j @B      ??!       J	?X??+????X??+???!?X??+???R      ??!       Z	?X??+????X??+???!?X??+???b      ??!       JGPUY?l\Rp?@b qgV???6@yu???7R@