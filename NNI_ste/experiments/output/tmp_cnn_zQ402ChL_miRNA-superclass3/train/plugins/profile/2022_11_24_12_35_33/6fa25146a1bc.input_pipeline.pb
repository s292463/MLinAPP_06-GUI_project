	?Ӂ???@?Ӂ???@!?Ӂ???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Ӂ???@B?????1øDk??A1]??a??I?R?o*R@rEagerKernelExecute 0*	????K??@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????"??!?w??6?P@)???!9???1l?i??D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?^f?(??!I?G5@){??v? ??1?@f P5@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map-\Va3???!??B?3@)ޮ??p??1??.#?\-@:Preprocessing2F
Iterator::Model???ۂ??!.?2?r#@)?-??ĥ?1[n`[+-@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?*8? ??!??j?D"@)?H?}??1p#C?@:Preprocessing2U
Iterator::Model::ParallelMapV2?uoEb???!|
?Yq@)?uoEb???1|
?Yq@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice'?Wʒ?!??`???@)'?Wʒ?1??`???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~k'JB??!??>?G@)pz?????1&o????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch!;oc?#??!??i%?z??)!;oc?#??1??i%?z??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'l?????!??,ߣQ@)??~??@?1????Z??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorp??-y?!??c????)p??-y?1??c????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangek???p?! $>
???)k???p?1 $>
???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate??g͏???!???S@)??^?S_?1??swe??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?7??w?S?!?g&Q%??)?7??w?S?1?g&Q%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 25.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?55.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?"#i?T@Q?ts[??1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B?????B?????!B?????      ??!       "	øDk??øDk??!øDk??*      ??!       2	1]??a??1]??a??!1]??a??:	?R?o*R@?R?o*R@!?R?o*R@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?"#i?T@y?ts[??1@