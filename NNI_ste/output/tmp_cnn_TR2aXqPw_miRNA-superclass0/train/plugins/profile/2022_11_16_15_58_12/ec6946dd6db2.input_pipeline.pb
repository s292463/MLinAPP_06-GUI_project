	R????$$@R????$$@!R????$$@	t-(n????t-(n????!t-(n????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLR????$$@o???????10?[w??@Ab????k??I?xwd??@YE|V??rEagerKernelExecute 0*	?K7?A f@2F
Iterator::Model?x\T????!a?W???D@)U?q7?֪?1x???=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatߌ??????!?c?<??7@)??[;Q??1????4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip} y?P???!??_oM@)ǟ?lXS??1C??o,@:Preprocessing2U
Iterator::Model::ParallelMapV2?Վ?u??!??@???&@)?Վ?u??1??@???&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceTȕz???!*?å??$@)Tȕz???1*?å??$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatem???{???!??w?S?2@)???????1??+??!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????{?!&]kE??@)?????{?1&]kE??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9d?bӢ?!?ˡ???4@)??I`sn?1NjP0? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?22.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9t-(n????IE?פ??C@QO???N?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o???????o???????!o???????      ??!       "	0?[w??@0?[w??@!0?[w??@*      ??!       2	b????k??b????k??!b????k??:	?xwd??@?xwd??@!?xwd??@B      ??!       J	E|V??E|V??!E|V??R      ??!       Z	E|V??E|V??!E|V??b      ??!       JGPUYt-(n????b qE?פ??C@yO???N?M@