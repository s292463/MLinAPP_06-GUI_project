	⬈??S@⬈??S@!⬈??S@	mfR!?@mfR!?@!mfR!?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL⬈??S@????*???1??X?_b@A?J?4??I????S@Y0??{???rEagerKernelExecute 0*	`??"??c@2F
Iterator::Model??!? ???!??????F@)_Pj??1>Bƿ??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\?J???!?iC">@)%?}?e???1/͏?^w9@:Preprocessing2U
Iterator::Model::ParallelMapV2?[??.???!?Ϟ`?*@)?[??.???1?Ϟ`?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?S???
??!5?8N/?(@)?S???
??15?8N/?(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?f??67??!-7BK@)?;ۤ???1S??>S?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?}?[?~??!??-?+{1@)?S:X????1?E?P?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??j?#?}?!8r?_@)??j?#?}?18r?_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\???4??!?n<3@)Ku/3ld?1?g????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9mfR!?@I?g?K??N@Qr?A??A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????*???????*???!????*???      ??!       "	??X?_b@??X?_b@!??X?_b@*      ??!       2	?J?4???J?4??!?J?4??:	????S@????S@!????S@B      ??!       J	0??{???0??{???!0??{???R      ??!       Z	0??{???0??{???!0??{???b      ??!       JGPUYmfR!?@b q?g?K??N@yr?A??A@