	n???@n???@!n???@	5?Xۑ?@5?Xۑ?@!5?Xۑ?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLn???@?B?_????1?hE???A??1?????I<K?P?@Y ?~?:p??rEagerKernelExecute 0*	|?G?]?@2U
Iterator::Model::ParallelMapV2??*?]g??!?A?J?S@)??*?]g??1?A?J?S@:Preprocessing2F
Iterator::Modelp^??jG??!'ؖ"??U@)??n??;??1UAP%J?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4??8??!?)??@)}x? #???1<?Yz?z@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?3?c?=??!TAV??@).v??2S??1???3ȗ @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???'??!?:?E?s??)???'??1?:?E?s??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??L0?k??!?>I??)@)1AG?Z??1@???;#??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???|?!~?AK#??)???|?1~?AK#??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?EaE??!%?rU??@)????u?f?1\??"n??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?52.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no95?Xۑ?@I"??]DS@QR$"N??5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?B?_?????B?_????!?B?_????      ??!       "	?hE????hE???!?hE???*      ??!       2	??1???????1?????!??1?????:	<K?P?@<K?P?@!<K?P?@B      ??!       J	 ?~?:p?? ?~?:p??! ?~?:p??R      ??!       Z	 ?~?:p?? ?~?:p??! ?~?:p??b      ??!       JGPUY5?Xۑ?@b q"??]DS@yR$"N??5@