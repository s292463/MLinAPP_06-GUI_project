	i6??`?@i6??`?@!i6??`?@	?3d?q???3d?q??!?3d?q??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLi6??`?@%??ID???1???im???Av?uŌ???I
pU
@Y?(??/??rEagerKernelExecute 0*	??Q?e@2F
Iterator::Model???X?y??!??@?J@)???K??1/G?"?%B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!?K?V??<@)???dȱ??1???x?(9@:Preprocessing2U
Iterator::Model::ParallelMapV2??DKO??!??x??/@)??DKO??1??x??/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????#??!p,>?@)????#??1p,>?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?x?a???!0?p?^?G@)?[v?؂?1`?#?.?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???!????!??5??=(@)?a?'֩??1?^P?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensord??uy?!?h?&?@)d??uy?1?h?&?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapl{?%9`??!2w??*+@)=????c?1?y?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?3d?q??I??1ٕS@Q5}I?3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	%??ID???%??ID???!%??ID???      ??!       "	???im??????im???!???im???*      ??!       2	v?uŌ???v?uŌ???!v?uŌ???:	
pU
@
pU
@!
pU
@B      ??!       J	?(??/???(??/??!?(??/??R      ??!       Z	?(??/???(??/??!?(??/??b      ??!       JGPUY?3d?q??b q??1ٕS@y5}I?3@