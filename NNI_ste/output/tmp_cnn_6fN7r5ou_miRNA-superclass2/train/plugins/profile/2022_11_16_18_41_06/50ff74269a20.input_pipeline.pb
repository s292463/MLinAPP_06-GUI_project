	??B??@??B??@!??B??@	?W???b@?W???b@!?W???b@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??B??@?f????1?F??@A???P??I@?J???@YL?[??.??rEagerKernelExecute 0*	?t??g@2F
Iterator::Model?dT8??!r?	 G@)kIG9?M??1NX\B?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?hE,??!?!??;@)?tA}˜??1?Ьwh7@:Preprocessing2U
Iterator::Model::ParallelMapV2?%s,流?!????(@)?%s,流?1????(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zips??P???!??4???J@)@?z??{??1???_?<&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice|???s??!?????"$@)|???s??1?????"$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateՕ??<???!?_?,@)?N?????1?̷@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn?8)?{|?!??ʮ?{@)n?8)?{|?1??ʮ?{@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptϺFˁ??!x??e??/@)??Y?rLf?13?v???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?38.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?W???b@Is???YN@Q?I?0B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f?????f????!?f????      ??!       "	?F??@?F??@!?F??@*      ??!       2	???P?????P??!???P??:	@?J???@@?J???@!@?J???@B      ??!       J	L?[??.??L?[??.??!L?[??.??R      ??!       Z	L?[??.??L?[??.??!L?[??.??b      ??!       JGPUY?W???b@b qs???YN@y?I?0B@