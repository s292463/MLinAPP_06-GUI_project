	???~~@???~~@!???~~@	fR???@fR???@!fR???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???~~@G?&???175?|N@Ap???$??I?A?f?@Y?n?UfJ??rEagerKernelExecute 0*	???S??a@2F
Iterator::Modelc?: ⮮?!"j?*?	E@)x'???1?@)??7:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Z_$????! N?Ƨ<@)?????1\??T??8@:Preprocessing2U
Iterator::Model::ParallelMapV2?H??? ??!n'?҂?/@)?H??? ??1n'?҂?/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?d??!?wr\?*@)?d??1?wr\?*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQL? 3??!?\??d5@)?s|?8c??1~A??? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipbI????!ޕ5??L@)D?l?????1???D?8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?N?6??v?!?5?ĳ?@)?N?6??v?1?5?ĳ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8????!|o??77@)??fHe?1o-qs/??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9fR???@I??(?]?P@Q??q??V=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	G?&???G?&???!G?&???      ??!       "	75?|N@75?|N@!75?|N@*      ??!       2	p???$??p???$??!p???$??:	?A?f?@?A?f?@!?A?f?@B      ??!       J	?n?UfJ???n?UfJ??!?n?UfJ??R      ??!       Z	?n?UfJ???n?UfJ??!?n?UfJ??b      ??!       JGPUYfR???@b q??(?]?P@y??q??V=@