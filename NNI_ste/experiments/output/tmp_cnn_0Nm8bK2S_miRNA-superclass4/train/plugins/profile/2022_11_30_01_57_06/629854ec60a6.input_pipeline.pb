	O#-??;%@O#-??;%@!O#-??;%@	???t@???t@!???t@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCO#-??;%@?p?????1?cyW= @Ia5??6???Y*?=%????rEagerKernelExecute 0*	??ʡEve@2F
Iterator::Model???ދ/??!^U0?<I@)s?SrN???1ңG?}=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatøDkE??!?ڜ??@)p??????1o{?6?:@:Preprocessing2U
Iterator::Model::ParallelMapV2?8K?r??!??b?e?4@)?8K?r??1??b?e?4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceAH0?[??!?&????@)AH0?[??1?&????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateAF@?#H??!X[am?5(@)ADj??4??1ڏ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6??ĵ?!????=?H@)kF????1?L?V?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???N~?!?????@)???N~?1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???0B??!b Гd?+@)<??k?g?1K(u3???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?16.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???t@I?(3p5@Q???WR?R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p??????p?????!?p?????      ??!       "	?cyW= @?cyW= @!?cyW= @*      ??!       2      ??!       :	a5??6???a5??6???!a5??6???B      ??!       J	*?=%????*?=%????!*?=%????R      ??!       Z	*?=%????*?=%????!*?=%????b      ??!       JGPUY???t@b q?(3p5@y???WR?R@