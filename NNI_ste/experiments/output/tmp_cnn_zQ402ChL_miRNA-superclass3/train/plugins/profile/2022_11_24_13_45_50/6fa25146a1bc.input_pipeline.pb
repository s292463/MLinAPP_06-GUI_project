	?k?}?:@?k?}?:@!?k?}?:@	?W?|%???W?|%??!?W?|%??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?k?}?:@??-Y???1	?%qVD??A??:?p??I=????@Y?ECƣT??rEagerKernelExecute 0*	??K7?Ie@2F
Iterator::Model)@̘???!û|??AM@)??y?):??1?ެi??D@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???ZӼ??!?? ??6@)Ҩ??6??1??O?{?2@:Preprocessing2U
Iterator::Model::ParallelMapV2?ej?!??!޹????0@)?ej?!??1޹????0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???0`Ʌ?! u,!??@)???0`Ʌ?1 u,!??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatej'?;??!???q?,(@))狽_??1J k?]@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#?~???!=D?O%?D@)v??ݰm??1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????L0|?!?D?#*@)????L0|?1?D?#*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?n.??'??!?D?+@)??:8؛h?19K?#59??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?W?|%??I?B4??S@QڮNf:4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-Y?????-Y???!??-Y???      ??!       "		?%qVD??	?%qVD??!	?%qVD??*      ??!       2	??:?p????:?p??!??:?p??:	=????@=????@!=????@B      ??!       J	?ECƣT???ECƣT??!?ECƣT??R      ??!       Z	?ECƣT???ECƣT??!?ECƣT??b      ??!       JGPUY?W?|%??b q?B4??S@yڮNf:4@